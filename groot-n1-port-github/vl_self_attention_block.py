"""
vl_self_attention_block.py — GR00TVLSelfAttnModel + NeuronGR00TVLSelfAttnWrapper

4-layer SelfAttentionTransformer for VL conditioning feature refinement.
Compiled via ModelBuilder with TP=8.

Architecture (from Isaac-GR00T SelfAttentionTransformer + diffusers BasicTransformerBlock):
  - 4 layers, each:
      norm1  = nn.LayerNorm(2048)          standard LayerNorm (elementwise_affine=True)
      attn1  = MultiHeadAttention(32 heads × 64 dim = 2048), bias=True
      norm3  = nn.LayerNorm(2048)
      ff     = FeedForward: GELU(2048→8192) + nn.Linear(8192→2048), bias=True
  - No RoPE (positional_embeddings=None)
  - No GQA (32 heads for Q, K, V)
  - Residual around both attention and FFN

Weight mapping from checkpoint:
  action_head.vl_self_attention.transformer_blocks.N.norm1.{weight,bias}
    → model.layers.N.norm1.{weight,bias}
  action_head.vl_self_attention.transformer_blocks.N.attn1.to_q.{weight,bias}
    → model.layers.N.attn.to_q.{weight,bias}
  action_head.vl_self_attention.transformer_blocks.N.attn1.to_k.{weight,bias}
    → model.layers.N.attn.to_k.{weight,bias}
  action_head.vl_self_attention.transformer_blocks.N.attn1.to_v.{weight,bias}
    → model.layers.N.attn.to_v.{weight,bias}
  action_head.vl_self_attention.transformer_blocks.N.attn1.to_out.0.{weight,bias}
    → model.layers.N.attn.to_out.{weight,bias}
  action_head.vl_self_attention.transformer_blocks.N.norm3.{weight,bias}
    → model.layers.N.norm3.{weight,bias}
  action_head.vl_self_attention.transformer_blocks.N.ff.net.0.proj.{weight,bias}
    → model.layers.N.ff_up.{weight,bias}
  action_head.vl_self_attention.transformer_blocks.N.ff.net.2.{weight,bias}
    → model.layers.N.ff_down.{weight,bias}

TP=8 split:
  to_q/to_k/to_v: ColumnParallelLinear (split output heads across 8 ranks)
  to_out/ff_down: RowParallelLinear (all-reduce partial sums)
  ff_up:          ColumnParallelLinear (split FFN intermediate)

KEY: FFN has bias on both projections. Do NOT use NeuronLlamaMLP — it drops bias silently.
Use ColumnParallelLinear/RowParallelLinear with bias=True directly.

CRITICAL: model is NOT constructed in __init__. Constructed in load_module() AFTER
ModelBuilder has initialized parallel_state(tp_degree=8). This ensures
ColumnParallelLinear/RowParallelLinear use real TP, not nn.Linear fallback.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
SKILLS_ROOT = os.path.join(ROOT, 'skills', 'scripts')
sys.path.insert(0, ROOT)
sys.path.insert(0, SKILLS_ROOT)
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

from config_constants import (
    BATCH_SIZE, NUM_CONDITIONING_TOKENS, TP_DEGREE_VLM,
    VL_SELF_ATTN_NUM_LAYERS, VL_SELF_ATTN_HIDDEN_SIZE,
    VL_SELF_ATTN_NUM_HEADS, VL_SELF_ATTN_HEAD_DIM,
    VL_SELF_ATTN_INTERMEDIATE_SIZE,
)

# ---------------------------------------------------------------------------
# NxDI parallel layers — fall back to nn.Linear when parallel_state absent
# ---------------------------------------------------------------------------
try:
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear, RowParallelLinear,
    )
    _PARALLEL_AVAILABLE = True
except ImportError:
    _PARALLEL_AVAILABLE = False

try:
    from neuronx_distributed.parallel_layers import parallel_state
    def _parallel_initialized():
        return parallel_state.model_parallel_is_initialized()
except ImportError:
    def _parallel_initialized():
        return False


def _col(in_f: int, out_f: int, bias: bool = True) -> nn.Module:
    """Column-parallel linear: split output dim. Fallback to nn.Linear on CPU."""
    if _PARALLEL_AVAILABLE and _parallel_initialized():
        return ColumnParallelLinear(in_f, out_f, bias=bias, gather_output=False)
    return nn.Linear(in_f, out_f, bias=bias)


def _row(in_f: int, out_f: int, bias: bool = True) -> nn.Module:
    """Row-parallel linear: split input dim + all-reduce. Fallback to nn.Linear on CPU."""
    if _PARALLEL_AVAILABLE and _parallel_initialized():
        return RowParallelLinear(in_f, out_f, bias=bias, input_is_parallel=True)
    return nn.Linear(in_f, out_f, bias=bias)


# ---------------------------------------------------------------------------
# Layer sub-modules (constructed inside load_module → parallel_state active)
# ---------------------------------------------------------------------------

class VLSelfAttnLayer(nn.Module):
    """
    Single SelfAttentionTransformer block.
    Matches diffusers BasicTransformerBlock with:
      - norm_type='layer_norm' (standard LayerNorm, no ada_norm)
      - positional_embeddings=None (no RoPE, no sinusoidal)
      - activation_fn='gelu-approximate' (GELU tanh approx)
      - attention_bias=True, ff_bias=True

    Constructed inside GR00TVLSelfAttnModel.__init__() which is called from
    load_module() — so parallel_state is active when this __init__ runs.
    """

    def __init__(self):
        super().__init__()
        H = VL_SELF_ATTN_HIDDEN_SIZE      # 2048
        I = VL_SELF_ATTN_INTERMEDIATE_SIZE  # 8192
        Nh = VL_SELF_ATTN_NUM_HEADS        # 32
        Dh = VL_SELF_ATTN_HEAD_DIM         # 64

        # Self-attention projections (TP split over heads)
        self.to_q   = _col(H, H,  bias=True)   # [2048, 2048]
        self.to_k   = _col(H, H,  bias=True)   # [2048, 2048]
        self.to_v   = _col(H, H,  bias=True)   # [2048, 2048]
        self.to_out = _row(H, H,  bias=True)   # [2048, 2048], all-reduce

        # LayerNorms (not split — scalar ops, no TP needed)
        self.norm1 = nn.LayerNorm(H, eps=1e-5, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(H, eps=1e-5, elementwise_affine=True)

        # FFN with bias: up (ColumnParallel) → GELU → down (RowParallel)
        self.ff_up   = _col(H, I, bias=True)   # [8192, 2048]
        self.ff_down = _row(I, H, bias=True)   # [2048, 8192], all-reduce

        # Store for reshape logic
        self.num_heads = Nh
        self.head_dim  = Dh
        self.scale     = Dh ** -0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        # --- Self-attention ---
        norm_hs = self.norm1(hidden_states)

        q = self.to_q(norm_hs)   # [B, T, inner_dim_local] (local if TP)
        k = self.to_k(norm_hs)
        v = self.to_v(norm_hs)

        # Reshape for multi-head attention
        # When ColumnParallelLinear is active: inner_dim_local = heads*head_dim / tp
        # When nn.Linear fallback: inner_dim_local = heads*head_dim
        local_inner = q.shape[-1]
        local_heads = local_inner // self.head_dim

        q = q.reshape(B, T, local_heads, self.head_dim).transpose(1, 2)  # [B, Nh_local, T, Dh]
        k = k.reshape(B, T, local_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, local_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, Nh_local, T, T]
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(hidden_states.dtype)

        attn_out = torch.matmul(attn_weights, v)             # [B, Nh_local, T, Dh]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, local_inner)

        # RowParallelLinear all-reduces across ranks here
        attn_out = self.to_out(attn_out)
        hidden_states = hidden_states + attn_out

        # --- FFN ---
        norm_hs = self.norm3(hidden_states)
        ff_out = self.ff_up(norm_hs)
        ff_out = F.gelu(ff_out, approximate='tanh')
        ff_out = self.ff_down(ff_out)
        hidden_states = hidden_states + ff_out

        return hidden_states


class GR00TVLSelfAttnModel(nn.Module):
    """
    4-layer SelfAttentionTransformer with TP=8 parallel layers.

    IMPORTANT: This class is constructed inside NeuronGR00TVLSelfAttnWrapper.load_module()
    so parallel_state is already active when __init__ runs.
    Do NOT instantiate this outside load_module() on a Trainium instance.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [VLSelfAttnLayer() for _ in range(VL_SELF_ATTN_NUM_LAYERS)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input/output: [B, NUM_CONDITIONING_TOKENS, 2048] BF16."""
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Wrapper for ModelBuilder compilation (TP=8)
# ---------------------------------------------------------------------------

class NeuronGR00TVLSelfAttnWrapper(nn.Module):
    """
    ModelWrapper-compatible wrapper for GR00TVLSelfAttnModel.

    Bypasses ModelWrapper.__init__ to avoid distributed state setup before
    parallel_state is initialized. Only nn.Module.__init__ is called.

    Pattern: same as NeuronGR00TDenoisingWrapper — lazy model init in load_module().
    """

    def __init__(self):
        nn.Module.__init__(self)   # bypass ModelWrapper.__init__
        self.model = None          # lazy — constructed in load_module()
        self._preload_sd = None    # stashed state dict before load_module()
        self.tag = 'vl_self_attn'

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Auto-init model for pre_compile_validate() on CPU.
        if self.model is None:
            self.load_module()
        return self.model(hidden_states)

    def input_generator(self):
        """Example inputs for XLA tracing. All zeros, static shapes."""
        return [(
            torch.zeros(BATCH_SIZE, NUM_CONDITIONING_TOKENS,
                        VL_SELF_ATTN_HIDDEN_SIZE, dtype=torch.bfloat16),
        )]

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """
        Accepts **kwargs so torch_neuronx's assign=True call doesn't raise TypeError.
        If model is not yet constructed, stash the state dict for load_module().
        """
        if self.model is None:
            self._preload_sd = state_dict
            # Return a dummy result (no parameters registered yet)
            return nn.Module.load_state_dict(self, {}, strict=False)
        return nn.Module.load_state_dict(self, state_dict, strict=strict, **kwargs)

    def load_module(self):
        """
        ModelBuilder calls this AFTER parallel_state(tp_degree=8) is active.
        Construct GR00TVLSelfAttnModel here so ColumnParallelLinear uses real TP=8.

        On CPU (pre_compile_validate): loads weights for correctness check.
        On Trainium (ModelBuilder): does NOT load weights (sharding handles this).
        """
        self.model = GR00TVLSelfAttnModel()
        on_trainium = _parallel_initialized()
        if self._preload_sd is not None and not on_trainium:
            # _preload_sd may have 'model.*' prefixed keys (wrapper-level).
            # GR00TVLSelfAttnModel expects keys without 'model.' prefix.
            model_pfx = 'model.'
            stripped_sd = {}
            for k, v in self._preload_sd.items():
                if k.startswith(model_pfx):
                    stripped_sd[k[len(model_pfx):]] = v
                else:
                    stripped_sd[k] = v
            self.model.load_state_dict(stripped_sd, strict=True)
        self.model = self.model.bfloat16().eval()

    def is_neuron(self) -> bool:
        """True if the compiled NEFF is loaded on Neuron hardware."""
        return self.model is not None and isinstance(self.model, torch.jit.ScriptModule)

    def get(self, bucket_rank=0, **kwargs):
        """ModelBuilder calls this to get (module, input_output_aliases)."""
        return self, {}


# ---------------------------------------------------------------------------
# Weight mapping: HF checkpoint → NeuronGR00TVLSelfAttnWrapper state_dict
# ---------------------------------------------------------------------------

def get_vl_self_attn_key_mapping() -> dict:
    """
    Map action_head.vl_self_attention.transformer_blocks.N.* keys
    to NeuronGR00TVLSelfAttnWrapper 'model.layers.N.*' keys.

    Direct key mapping — no weight merging or splitting needed.
    """
    pfx = 'action_head.vl_self_attention.'
    mapping = {}
    for i in range(VL_SELF_ATTN_NUM_LAYERS):
        mapping.update({
            # LayerNorms
            f'{pfx}transformer_blocks.{i}.norm1.weight':           f'model.layers.{i}.norm1.weight',
            f'{pfx}transformer_blocks.{i}.norm1.bias':             f'model.layers.{i}.norm1.bias',
            f'{pfx}transformer_blocks.{i}.norm3.weight':           f'model.layers.{i}.norm3.weight',
            f'{pfx}transformer_blocks.{i}.norm3.bias':             f'model.layers.{i}.norm3.bias',
            # Attention Q/K/V projections
            f'{pfx}transformer_blocks.{i}.attn1.to_q.weight':      f'model.layers.{i}.to_q.weight',
            f'{pfx}transformer_blocks.{i}.attn1.to_q.bias':        f'model.layers.{i}.to_q.bias',
            f'{pfx}transformer_blocks.{i}.attn1.to_k.weight':      f'model.layers.{i}.to_k.weight',
            f'{pfx}transformer_blocks.{i}.attn1.to_k.bias':        f'model.layers.{i}.to_k.bias',
            f'{pfx}transformer_blocks.{i}.attn1.to_v.weight':      f'model.layers.{i}.to_v.weight',
            f'{pfx}transformer_blocks.{i}.attn1.to_v.bias':        f'model.layers.{i}.to_v.bias',
            # Attention output projection
            f'{pfx}transformer_blocks.{i}.attn1.to_out.0.weight':  f'model.layers.{i}.to_out.weight',
            f'{pfx}transformer_blocks.{i}.attn1.to_out.0.bias':    f'model.layers.{i}.to_out.bias',
            # FFN: net.0.proj = up projection, net.2 = down projection
            f'{pfx}transformer_blocks.{i}.ff.net.0.proj.weight':   f'model.layers.{i}.ff_up.weight',
            f'{pfx}transformer_blocks.{i}.ff.net.0.proj.bias':     f'model.layers.{i}.ff_up.bias',
            f'{pfx}transformer_blocks.{i}.ff.net.2.weight':        f'model.layers.{i}.ff_down.weight',
            f'{pfx}transformer_blocks.{i}.ff.net.2.bias':          f'model.layers.{i}.ff_down.bias',
        })
    return mapping


def load_vl_self_attn_state_dict(hf_sd: dict) -> dict:
    """
    Extract VL self-attention weights from full GR00T checkpoint.
    Returns dict with keys matching NeuronGR00TVLSelfAttnWrapper state_dict.
    """
    mapping = get_vl_self_attn_key_mapping()
    out, missing = {}, []
    for hf_key, neuron_key in mapping.items():
        if hf_key in hf_sd:
            out[neuron_key] = hf_sd[hf_key]
        else:
            missing.append(hf_key)
    if missing:
        raise KeyError(
            f'Missing {len(missing)} VL self-attn checkpoint keys, e.g. {missing[:3]}'
        )
    return out


def load_vl_self_attn_model_state_dict(hf_sd: dict) -> dict:
    """
    Extract VL self-attention weights for GR00TVLSelfAttnModel directly.
    Strips 'model.' prefix (keys start with 'layers.N.*').
    Use this when loading into GR00TVLSelfAttnModel for unit tests.
    """
    wrapper_sd = load_vl_self_attn_state_dict(hf_sd)
    return {k.removeprefix('model.'): v for k, v in wrapper_sd.items()}
