"""
VLM Backbone Block for SmolVLA port to AWS Trainium (NxDI).

Implements the SmolLM2 prefix pass that produces conditioning KV tokens
for the cross-attention expert layers.

Input:  prefix_embs [B, 113, 960] BF16
Output: conditioning_tokens [B, 113, 5120] BF16
"""
import sys
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    _PARALLEL_AVAILABLE = True
except ImportError:
    _PARALLEL_AVAILABLE = False


def _is_parallel():
    return _PARALLEL_AVAILABLE and parallel_state.model_parallel_is_initialized()


def _tp_group():
    return parallel_state.get_tensor_model_parallel_group() if _is_parallel() else None

from config_constants import (
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
    HEAD_DIM,
    INTERMEDIATE_SIZE,
    NUM_HIDDEN_LAYERS,
    RMS_NORM_EPS,
    VLM_KV_DIM,
    PREFIX_SEQ_LEN,
    CONDITIONING_HIDDEN_SIZE,
    EXPERT_NUM_CROSS_ATTN_LAYERS,  # = 8
)


# ---------------------------------------------------------------------------
# Custom RoPE — uses pre-computed sin/cos tables to avoid duplicate constant
# nodes in the traced graph (torch.arange / timescale created once per forward
# across 16 layers generates a merged constant with >255-char filename).
# Tables are registered as buffers in NeuronVLMBackboneBlock.__init__.
# ---------------------------------------------------------------------------
def apply_rope_tables(x, sin_table, cos_table):
    """
    Apply RoPE using pre-computed sin/cos tables.

    x:         [B, L, H, D]
    sin_table: [1, L, 1, D//2]  float32
    cos_table: [1, L, 1, D//2]  float32

    Returns [B, L, H, D] in original dtype.
    """
    d_half = x.shape[-1] // 2
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x1, x2 = x[..., :d_half], x[..., d_half:]
    out = torch.cat([x1 * cos_table - x2 * sin_table,
                     x2 * cos_table + x1 * sin_table], dim=-1)
    return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class SmolRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * x).to(orig_dtype)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------
class SmolMLP(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE):
        super().__init__()
        if _is_parallel():
            g = _tp_group()
            self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False, gather_output=False, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.up_proj   = ColumnParallelLinear(hidden_size, intermediate_size, bias=False, gather_output=False, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.down_proj = RowParallelLinear(intermediate_size, hidden_size,    bias=False, input_is_parallel=True,  dtype=torch.bfloat16, tensor_model_parallel_group=g)
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class SmolSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim

        if _is_parallel():
            g = _tp_group()
            self.q_proj = ColumnParallelLinear(hidden_size, num_attention_heads * head_dim, bias=False, gather_output=True, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.k_proj = ColumnParallelLinear(hidden_size, num_key_value_heads * head_dim, bias=False, gather_output=True, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.v_proj = ColumnParallelLinear(hidden_size, num_key_value_heads * head_dim, bias=False, gather_output=True, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.o_proj = RowParallelLinear(num_attention_heads * head_dim, hidden_size,    bias=False, input_is_parallel=False, dtype=torch.bfloat16, tensor_model_parallel_group=g)
        else:
            self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask, rope_sin, rope_cos):
        """
        hidden_states: [B, L, hidden_size]
        attention_mask: [B, L, L] bool (True = attend)
        rope_sin: [1, L, 1, head_dim//2] float32  (pre-computed, registered buffer)
        rope_cos: [1, L, 1, head_dim//2] float32  (pre-computed, registered buffer)

        Returns:
            attn_output: [B, L, hidden_size]
            key_states: [B, L, num_key_value_heads, head_dim]  (after RoPE)
            value_states: [B, L, num_key_value_heads, head_dim]
        """
        B, L, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(B, L, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_key_value_heads, self.head_dim)

        # Apply RoPE to Q and K using pre-computed tables (avoids duplicate constant nodes)
        q = apply_rope_tables(q, rope_sin, rope_cos)
        k = apply_rope_tables(k, rope_sin, rope_cos)

        # GQA expansion — replicate logic from eager_attention_forward
        num_key_value_groups = self.num_key_value_groups

        # k: [B, L, num_kv_heads, head_dim] → expand → [B, L, num_heads, head_dim]
        k_expanded = k[:, :, :, None, :].expand(
            B, L, self.num_key_value_heads, num_key_value_groups, self.head_dim
        ).reshape(B, L, self.num_attention_heads, self.head_dim)

        v_expanded = v[:, :, :, None, :].expand(
            B, L, self.num_key_value_heads, num_key_value_groups, self.head_dim
        ).reshape(B, L, self.num_attention_heads, self.head_dim)

        # Attention computation (upcast to float32 per source)
        q_f = q.to(torch.float32).transpose(1, 2)        # [B, num_heads, L, head_dim]
        k_f = k_expanded.to(torch.float32).transpose(1, 2)  # [B, num_heads, L, head_dim]
        v_f = v_expanded.transpose(1, 2)                    # [B, num_heads, L, head_dim]

        attn_weights = torch.matmul(q_f, k_f.transpose(2, 3)) * (self.head_dim ** -0.5)
        attn_weights = attn_weights.to(torch.float32)

        big_neg = torch.finfo(attn_weights.dtype).min
        # attention_mask: [B, L, L] bool → need [B, 1, L, L] for broadcast
        attn_weights = torch.where(attention_mask[:, None, :, :], attn_weights, big_neg)
        probs = F.softmax(attn_weights, dim=-1).to(v_f.dtype)

        # [B, num_heads, L, head_dim]
        attn_out = torch.matmul(probs, v_f)
        # [B, L, num_heads * head_dim]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, self.num_attention_heads * self.head_dim)

        out = self.o_proj(attn_out)

        # Return output, plus K and V (after RoPE, shapes [B, L, num_kv_heads, head_dim])
        return out, k, v


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------
class SmolDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=INTERMEDIATE_SIZE,
        rms_norm_eps=RMS_NORM_EPS,
    ):
        super().__init__()
        self.input_layernorm = SmolRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = SmolSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        self.post_attention_layernorm = SmolRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SmolMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, hidden_states, attention_mask, rope_sin, rope_cos):
        """
        Returns:
            hidden_states: [B, L, hidden_size]
            k: [B, L, num_kv_heads, head_dim]
            v: [B, L, num_kv_heads, head_dim]
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, k, v = self.self_attn(hidden_states, attention_mask, rope_sin, rope_cos)
        hidden_states = attn_out + residual

        after_first_residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + after_first_residual

        return hidden_states, k, v


# ---------------------------------------------------------------------------
# Main NeuronVLMBackboneBlock
# ---------------------------------------------------------------------------
class NeuronVLMBackboneBlock(nn.Module):
    """
    SmolLM2 16-layer prefix pass that produces conditioning KV tokens
    for the 8 cross-attention expert layers (odd layers 1,3,5,7,9,11,13,15).

    Input:  prefix_embs   [B, PREFIX_SEQ_LEN, HIDDEN_SIZE] BF16
            attention_mask [B, PREFIX_SEQ_LEN, PREFIX_SEQ_LEN] bool
            position_ids   [B, PREFIX_SEQ_LEN] long

    Output: conditioning [B, PREFIX_SEQ_LEN, CONDITIONING_HIDDEN_SIZE] BF16
            = [B, 113, 5120]
    """

    def __init__(self, config=None):
        super().__init__()
        self.layers = nn.ModuleList([
            SmolDecoderLayer() for _ in range(NUM_HIDDEN_LAYERS)
        ])
        self.norm = SmolRMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)

        # Pre-compute RoPE sin/cos for positions 0..PREFIX_SEQ_LEN-1.
        # Registered as buffers so they are constants in the traced graph —
        # avoids torch.arange/timescale being re-created 16 times per forward,
        # which causes neuronx-cc errno 36 (constant filename > 255 chars).
        d_half = HEAD_DIM // 2
        positions = torch.arange(PREFIX_SEQ_LEN, dtype=torch.float32)
        freq_exp = (2.0 / HEAD_DIM) * torch.arange(d_half, dtype=torch.float32)
        timescale = 10_000.0 ** freq_exp
        radians = positions[:, None] / timescale[None, :]          # [L, d_half]
        self.register_buffer('rope_sin', torch.sin(radians)[None, :, None, :])  # [1, L, 1, d_half]
        self.register_buffer('rope_cos', torch.cos(radians)[None, :, None, :])

    def forward(self, prefix_embs, attention_mask, position_ids):
        """
        prefix_embs:    [B, L, HIDDEN_SIZE]
        attention_mask: [B, L, L] bool
        position_ids:   [B, L] long

        Returns: conditioning [B, L, CONDITIONING_HIDDEN_SIZE]
        """
        B, L, _ = prefix_embs.shape
        wdtype = self.layers[0].self_attn.q_proj.weight.dtype
        hidden_states = prefix_embs.to(wdtype)

        # Collect KV from all 16 layers
        k_list = []
        v_list = []

        for layer in self.layers:
            hidden_states, k, v = layer(hidden_states, attention_mask, self.rope_sin, self.rope_cos)
            k_list.append(k)
            v_list.append(v)

        # Pack KV from ODD layers only: 1, 3, 5, 7, 9, 11, 13, 15
        # (0-indexed, so odd indices)
        odd_k = [k_list[i] for i in range(1, NUM_HIDDEN_LAYERS, 2)]   # 8 layers
        odd_v = [v_list[i] for i in range(1, NUM_HIDDEN_LAYERS, 2)]   # 8 layers

        # Interleave K and V pairs into [B, L, 5120]:
        #   [k0, v0, k1, v1, ..., k7, v7] concatenated on last dim.
        # Build via sequential 2-input cats to keep each node's parent count = 2.
        # torch.cat(16 inputs) generates a constant node whose filename includes
        # all 16 parent IDs (>255 chars, crashes neuronx-cc with errno 36).
        conditioning = odd_k[0].reshape(B, L, VLM_KV_DIM)
        conditioning = torch.cat([conditioning, odd_v[0].reshape(B, L, VLM_KV_DIM)], dim=-1)
        for k, v in zip(odd_k[1:], odd_v[1:]):
            conditioning = torch.cat([conditioning, k.reshape(B, L, VLM_KV_DIM)], dim=-1)
            conditioning = torch.cat([conditioning, v.reshape(B, L, VLM_KV_DIM)], dim=-1)

        return conditioning


# ---------------------------------------------------------------------------
# PyTorch Reference (anti-cheat: uses actual SmolVLMWithExpertModel)
# ---------------------------------------------------------------------------
class SmolVLMBackboneReference(nn.Module):
    """
    Reference implementation using the installed lerobot SmolVLMWithExpertModel.
    Wraps the VLM-only prefix forward and packs KV into [B, 113, 5120].

    Accepts a SINGLE tensor (prefix_embs) per the test harness convention.
    """

    def __init__(self):
        super().__init__()
        from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
        self.model = SmolVLMWithExpertModel(
            load_vlm_weights=False,
            num_vlm_layers=NUM_HIDDEN_LAYERS,
        )
        # Set eager attention implementation
        text_model = self.model.get_vlm_model().text_model
        text_model._attn_implementation = "eager"

    def forward(self, prefix_embs):
        """
        prefix_embs: [B, PREFIX_SEQ_LEN, HIDDEN_SIZE]
        Returns: conditioning [B, PREFIX_SEQ_LEN, CONDITIONING_HIDDEN_SIZE]
        """
        B, L, _ = prefix_embs.shape
        device = prefix_embs.device

        # Construct position_ids
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # All-True causal mask (prefix is non-causal — attend everywhere)
        attention_mask = torch.ones(B, L, L, dtype=torch.bool, device=device)

        # Run VLM-only forward (inputs_embeds=[prefix_embs, None])
        _, past_key_values = self.model.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        # past_key_values is a dict {layer_idx: {"key_states": ..., "value_states": ...}}
        # key_states shape: [B, L, num_kv_heads, head_dim]
        # Pack odd layers (1, 3, 5, 7, 9, 11, 13, 15)
        k_flat = []
        v_flat = []
        for l_idx in range(1, NUM_HIDDEN_LAYERS, 2):
            k = past_key_values[l_idx]["key_states"]   # [B, L, 5, 64]
            v = past_key_values[l_idx]["value_states"]  # [B, L, 5, 64]
            k_flat.append(k.reshape(B, L, VLM_KV_DIM))
            v_flat.append(v.reshape(B, L, VLM_KV_DIM))

        k_stacked = torch.stack(k_flat, dim=2)   # [B, L, 8, 320]
        v_stacked = torch.stack(v_flat, dim=2)   # [B, L, 8, 320]
        kvstack = torch.stack([k_stacked, v_stacked], dim=3)  # [B, L, 8, 2, 320]
        conditioning = kvstack.reshape(B, L, CONDITIONING_HIDDEN_SIZE)

        return conditioning
