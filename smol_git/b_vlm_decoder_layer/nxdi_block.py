"""
NxDI VLM Text Decoder Layer for SmolVLM2-500M (Llama GQA).

Translates a single Llama decoder layer into a Neuron/NxDI-compatible module.
Prefix pass only (fill_kv_cache=True; no token-by-token decoding).

Input:
  hidden_states [1, 241, 960] bf16
  attention_mask [1, 241, 241] bool (True=attend)
  position_ids [1, 241] int64

Output tuple:
  hidden_states [1, 241, 960] bf16
  key_states [1, 241, 5, 64] bf16
  value_states [1, 241, 5, 64] bf16
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
    HAS_NXD = True
except Exception:
    HAS_NXD = False
    RotaryEmbedding = None  # replaced by _FallbackRoPE below when HAS_NXD is False


# ---------------------------------------------------------------------------
# Helper: apply_rotary_pos_emb compatible with NxDI RotaryEmbedding output
# ---------------------------------------------------------------------------

def _rotate_half(x):
    """Rotate the last dimension by half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    """
    q, k: [B, num_heads, S, head_dim]
    cos, sin: [B, S, head_dim] (from NxDI RotaryEmbedding)
    """
    # Unsqueeze to broadcast over heads: [B, 1, S, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


# ---------------------------------------------------------------------------
# CPU fallback RoPE (same interface as NxDI RotaryEmbedding)
# ---------------------------------------------------------------------------

class _FallbackRoPE(nn.Module):
    """Minimal RoPE matching NxDI RotaryEmbedding output signature."""

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 100000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # x: [B, nH, S, hd] — used only for device/dtype
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)          # [B, S, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)       # [B, S, dim]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


# ---------------------------------------------------------------------------
# Norm helper
# ---------------------------------------------------------------------------

def _make_rms_norm(hidden_size, eps):
    if HAS_NXD and parallel_state.model_parallel_is_initialized():
        from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
        return CustomRMSNorm(hidden_size, eps)
    else:
        return nn.RMSNorm(hidden_size, eps=eps)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class NeuronSmolVLMDecoderAttention(nn.Module):
    """
    GQA attention: 15 Q-heads, 5 KV-heads, head_dim=64.
    Uses ColumnParallelLinear/RowParallelLinear when TP is initialized.
    """

    def __init__(
        self,
        hidden_size: int = 960,
        num_attention_heads: int = 15,
        num_key_value_heads: int = 5,
        head_dim: int = 64,
        max_position_embeddings: int = 8192,
        rope_theta: float = 100000.0,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.scale = head_dim ** -0.5

        q_out = num_attention_heads * head_dim   # 960
        kv_out = num_key_value_heads * head_dim  # 320

        if HAS_NXD and parallel_state.model_parallel_is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()
            kwargs = dict(bias=attention_bias, gather_output=False, dtype=torch.bfloat16,
                          sequence_parallel_enabled=False, sequence_dimension=None,
                          tensor_model_parallel_group=tp_group)
            self.q_proj = ColumnParallelLinear(hidden_size, q_out, **kwargs)
            self.k_proj = ColumnParallelLinear(hidden_size, kv_out, **kwargs)
            self.v_proj = ColumnParallelLinear(hidden_size, kv_out, **kwargs)
            self.o_proj = RowParallelLinear(
                q_out, hidden_size,
                bias=attention_bias,
                input_is_parallel=True,
                dtype=torch.bfloat16,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.q_proj = nn.Linear(hidden_size, q_out, bias=attention_bias)
            self.k_proj = nn.Linear(hidden_size, kv_out, bias=attention_bias)
            self.v_proj = nn.Linear(hidden_size, kv_out, bias=attention_bias)
            self.o_proj = nn.Linear(q_out, hidden_size, bias=attention_bias)

        rope_cls = RotaryEmbedding if HAS_NXD else _FallbackRoPE
        self.rotary_emb = rope_cls(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(self, hidden_states, attention_mask, position_ids):
        """
        hidden_states: [B, S, hidden_size]
        attention_mask: [B, S, S] bool (True=attend, False=masked)
        position_ids: [B, S] int64

        Returns: (attn_output [B, S, hidden], key_states [B, S, nKV, D], value_states [B, S, nKV, D])
        """
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states)  # [B, S, nQ*D]
        k = self.k_proj(hidden_states)  # [B, S, nKV*D]
        v = self.v_proj(hidden_states)  # [B, S, nKV*D]

        # Reshape: [B, S, nH, D] -> transpose to [B, nH, S, D]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE: RotaryEmbedding expects x in [B, nH, S, D] format, position_ids [B, S]
        cos, sin = self.rotary_emb(q, position_ids)  # cos/sin: [B, S, D]
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Store KV before GQA expansion (for output)
        # key_states/value_states: [B, nKV, S, D] -> will transpose to [B, S, nKV, D]
        key_states_out = k.transpose(1, 2)    # [B, S, nKV, D]
        value_states_out = v.transpose(1, 2)  # [B, S, nKV, D]

        # GQA: expand KV heads to match Q heads
        # k: [B, nKV, S, D] -> [B, nQ, S, D]
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        # q, k, v: [B, nQ, S, D]
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, nQ, S, S]

        # attention_mask: [B, S, S] bool; True=attend, False=masked
        # Expand to [B, 1, S, S] for broadcasting
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)  # [B, 1, S, S]
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)  # [B, nQ, S, D]

        # Merge heads: [B, nQ, S, D] -> [B, S, nQ*D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, key_states_out, value_states_out


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class NeuronSmolVLMDecoderMLP(nn.Module):
    """
    Gated SiLU MLP: hidden_size=960, intermediate_size=2560.
    gate_proj + up_proj -> SiLU(gate) * up -> down_proj
    """

    def __init__(
        self,
        hidden_size: int = 960,
        intermediate_size: int = 2560,
        mlp_bias: bool = False,
    ):
        super().__init__()
        self.act_fn = F.silu

        if HAS_NXD and parallel_state.model_parallel_is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()
            col_kwargs = dict(bias=mlp_bias, gather_output=False, dtype=torch.bfloat16,
                              sequence_parallel_enabled=False, sequence_dimension=None,
                              tensor_model_parallel_group=tp_group)
            self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, **col_kwargs)
            self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, **col_kwargs)
            self.down_proj = RowParallelLinear(
                intermediate_size, hidden_size,
                bias=mlp_bias,
                input_is_parallel=True,
                dtype=torch.bfloat16,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class NeuronSmolVLMDecoderLayer(nn.Module):
    """
    Single Llama-style decoder layer for SmolVLM2-500M text model.

    Input:
      hidden_states [B, S, 960] bf16
      attention_mask [B, S, S] bool
      position_ids [B, S] int64

    Output:
      (hidden_states [B, S, 960], key_states [B, S, 5, 64], value_states [B, S, 5, 64])
    """

    def __init__(
        self,
        hidden_size: int = 960,
        intermediate_size: int = 2560,
        num_attention_heads: int = 15,
        num_key_value_heads: int = 5,
        head_dim: int = 64,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 100000.0,
        attention_bias: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.input_layernorm = _make_rms_norm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = _make_rms_norm(hidden_size, rms_norm_eps)

        self.self_attn = NeuronSmolVLMDecoderAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
        )

        self.mlp = NeuronSmolVLMDecoderMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, hidden_states, attention_mask, position_ids):
        # Pre-norm + attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, key_states, value_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + attn_out

        # Pre-norm + MLP + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, key_states, value_states
