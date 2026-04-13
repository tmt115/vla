"""
NxDI Expert Layer for SmolVLA Action Expert (Llama-based, cross-attention variant).

Translates the SmolVLA action expert LlamaDecoderLayer into a Neuron/NxDI-compatible
module. Supports two modes:
  - Self-attention (is_self_attn_layer=True):  standard GQA over 50 expert tokens
  - Cross-attention (is_self_attn_layer=False): expert Q attends to VLM KV cache

Architecture parameters (derived from SmolVLM2 text_config × 0.75):
  expert_hidden_size     = 720
  expert_intermediate    = 1536
  num_attention_heads    = 15
  num_key_value_heads    = 5
  head_dim               = 64
  vlm_kv_dim             = 320  (5 × 64, for cross-attn K/V input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    HAS_NXD = True
except Exception:
    HAS_NXD = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard rotate-half used in Llama RoPE (transformers convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _RoPE(nn.Module):
    """Minimal RoPE matching transformers LlamaRotaryEmbedding (no scaling)."""

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 100000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # x: [B, heads, S, head_dim]  (used only for device/dtype)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _apply_rotary(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to q only.  cos/sin: [B, S, head_dim]."""
    cos = cos.unsqueeze(1)   # [B, 1, S, head_dim]
    sin = sin.unsqueeze(1)   # [B, 1, S, head_dim]
    return (q * cos) + (_rotate_half(q) * sin)


# ---------------------------------------------------------------------------
# Parallel-linear factory helpers
# ---------------------------------------------------------------------------

def _col_linear(in_f: int, out_f: int) -> nn.Module:
    if HAS_NXD and parallel_state.model_parallel_is_initialized():
        tp_group = parallel_state.get_tensor_model_parallel_group()
        return ColumnParallelLinear(
            in_f, out_f,
            bias=False,
            gather_output=False,
            dtype=torch.bfloat16,

            sequence_parallel_enabled=False,
            sequence_dimension=None,
            tensor_model_parallel_group=tp_group,
        )
    return nn.Linear(in_f, out_f, bias=False)


def _row_linear(in_f: int, out_f: int) -> nn.Module:
    if HAS_NXD and parallel_state.model_parallel_is_initialized():
        tp_group = parallel_state.get_tensor_model_parallel_group()
        return RowParallelLinear(
            in_f, out_f,
            bias=False,
            input_is_parallel=True,
            dtype=torch.bfloat16,

            sequence_parallel_enabled=False,
            sequence_dimension=None,
            tensor_model_parallel_group=tp_group,
        )
    return nn.Linear(in_f, out_f, bias=False)


# ---------------------------------------------------------------------------
# Expert Attention
# ---------------------------------------------------------------------------

class NeuronSmolVLAExpertAttention(nn.Module):
    """
    GQA attention for the SmolVLA action expert.

    Self-attn mode  (is_cross_attn=False):
        Q, K, V all come from expert_hidden [B, 50, 720].
    Cross-attn mode (is_cross_attn=True):
        Q from expert_hidden; K/V from VLM KV tensors [B, 241, 5, 64].
    """

    NUM_HEADS    = 15   # expert num_attention_heads
    NUM_KV_HEADS = 5    # expert num_key_value_heads
    HEAD_DIM     = 64
    EXPERT_HIDDEN = 720
    VLM_KV_DIM   = 320  # 5 * 64

    def __init__(self, is_cross_attn: bool = False):
        super().__init__()
        self.is_cross_attn = is_cross_attn
        nh, nkv, hd = self.NUM_HEADS, self.NUM_KV_HEADS, self.HEAD_DIM
        eh, kd = self.EXPERT_HIDDEN, self.VLM_KV_DIM

        # Q: expert_hidden → num_heads * head_dim
        self.q_proj = _col_linear(eh, nh * hd)          # 720 → 960

        # K/V:
        #   self-attn:  expert_hidden  → num_kv_heads * head_dim (720 → 320)
        #   cross-attn: vlm_kv_dim     → num_kv_heads * head_dim (320 → 320)
        kv_in = kd if is_cross_attn else eh
        self.k_proj = _col_linear(kv_in, nkv * hd)
        self.v_proj = _col_linear(kv_in, nkv * hd)

        # O: num_heads * head_dim → expert_hidden  (960 → 720)
        self.o_proj = _row_linear(nh * hd, eh)

        self.rotary_emb = _RoPE(hd, max_position_embeddings=8192, base=100000.0)
        self.scale = hd ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, S_q, 720]
        attention_mask: torch.Tensor,          # [B, S_q, S_kv]  bool
        position_ids: torch.Tensor,            # [B, S_q]
        kv_key:   torch.Tensor | None = None,  # [B, S_kv, NUM_KV_HEADS, HEAD_DIM]
        kv_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S_q, _ = hidden_states.shape
        nh, nkv, hd = self.NUM_HEADS, self.NUM_KV_HEADS, self.HEAD_DIM

        # ---- Q -----------------------------------------------------------
        q = self.q_proj(hidden_states).view(B, S_q, nh, hd)  # [B, S_q, 15, 64]

        # ---- K/V ---------------------------------------------------------
        if not self.is_cross_attn:
            k = self.k_proj(hidden_states).view(B, S_q, nkv, hd)
            v = self.v_proj(hidden_states).view(B, S_q, nkv, hd)
        else:
            S_kv = kv_key.shape[1]
            # Flatten VLM KV from [B, S_kv, nkv, hd] → [B, S_kv, 320]
            kv_key_flat   = kv_key.reshape(B, S_kv, self.VLM_KV_DIM)
            kv_value_flat = kv_value.reshape(B, S_kv, self.VLM_KV_DIM)
            k = self.k_proj(kv_key_flat).view(B, S_kv, nkv, hd)
            v = self.v_proj(kv_value_flat).view(B, S_kv, nkv, hd)

        # ---- RoPE --------------------------------------------------------
        # Transpose to [B, heads, S, head_dim] for rotary_emb
        q_t = q.transpose(1, 2)  # [B, 15, S_q, 64]
        cos, sin = self.rotary_emb(q_t, position_ids)     # [B, S_q, 64]
        q_t = _apply_rotary(q_t, cos, sin)

        if not self.is_cross_attn:
            # Apply RoPE to K as well (same position_ids)
            k_t = k.transpose(1, 2)  # [B, 5, S_q, 64]
            cos_k, sin_k = self.rotary_emb(k_t, position_ids)
            k_t = _apply_rotary(k_t, cos_k, sin_k)
        else:
            # Cross-attention: K is VLM KV, no RoPE applied to K
            k_t = k.transpose(1, 2)  # [B, 5, S_kv, 64]

        v_t = v.transpose(1, 2)  # [B, 5, S_kv, 64]

        # ---- GQA expand --------------------------------------------------
        # Expand KV heads 5 → 15 (repeat each KV head 3 times)
        n_rep = nh // nkv  # 3
        k_t = k_t.unsqueeze(2).expand(B, nkv, n_rep, -1, hd).reshape(B, nh, -1, hd)
        v_t = v_t.unsqueeze(2).expand(B, nkv, n_rep, -1, hd).reshape(B, nh, -1, hd)

        # ---- Attention scores --------------------------------------------
        # q_t: [B, 15, S_q, 64]   k_t: [B, 15, S_kv, 64]
        scores = torch.matmul(q_t, k_t.transpose(2, 3)) * self.scale  # [B, 15, S_q, S_kv]

        # attention_mask: [B, S_q, S_kv] bool (True = attend)
        mask = attention_mask.unsqueeze(1)  # [B, 1, S_q, S_kv]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        probs = F.softmax(scores.float(), dim=-1).to(hidden_states.dtype)

        # ---- Weighted sum ------------------------------------------------
        attn_out = torch.matmul(probs, v_t)              # [B, 15, S_q, 64]
        attn_out = attn_out.transpose(1, 2).reshape(B, S_q, nh * hd)  # [B, S_q, 960]
        return self.o_proj(attn_out)                      # [B, S_q, 720]


# ---------------------------------------------------------------------------
# Expert MLP  (SwiGLU, hidden=720, intermediate=1536)
# ---------------------------------------------------------------------------

class NeuronSmolVLAExpertMLP(nn.Module):
    def __init__(self):
        super().__init__()
        hidden, intermediate = 720, 2048
        self.gate_proj = _col_linear(hidden, intermediate)
        self.up_proj   = _col_linear(hidden, intermediate)
        self.down_proj = _row_linear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Expert Decoder Layer
# ---------------------------------------------------------------------------

class NeuronSmolVLAExpertLayer(nn.Module):
    """
    Single SmolVLA action-expert decoder layer.

    Args:
        config: InferenceConfig (accepted for API compatibility, not used internally).
        is_self_attn_layer: True for even-indexed layers (self-attention only),
                            False for odd-indexed layers (cross-attention).
    """

    def __init__(self, config=None, is_self_attn_layer: bool = True):
        super().__init__()
        self.is_self_attn_layer = is_self_attn_layer

        self.input_layernorm         = nn.RMSNorm(720, eps=1e-5)
        self.post_attention_layernorm = nn.RMSNorm(720, eps=1e-5)

        self.self_attn = NeuronSmolVLAExpertAttention(is_cross_attn=not is_self_attn_layer)
        self.mlp       = NeuronSmolVLAExpertMLP()

    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, 50, 720]  bf16
        attention_mask: torch.Tensor,          # [B, 50, S_kv] bool
        position_ids: torch.Tensor,            # [B, 50] int64
        kv_key:   torch.Tensor | None = None,  # [B, 241, 5, 64] bf16  (cross-attn only)
        kv_value: torch.Tensor | None = None,  # [B, 241, 5, 64] bf16  (cross-attn only)
    ) -> torch.Tensor:
        # --- Attention sub-layer ---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask, position_ids, kv_key, kv_value
        )
        hidden_states = residual + hidden_states

        # --- MLP sub-layer ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
