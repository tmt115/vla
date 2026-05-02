"""
dit_block.py — GR00TDiTModel + NeuronGR00TDenoisingWrapper + NeuronGR00TActionHead

Translates AlternateVLDiT (32-layer flow-matching DiT) to Trainium TP=8.

Architecture split:
  GR00TDiTModel              — pure nn.Module, ColumnParallelLinear/RowParallelLinear
                               Falls back to nn.Linear on CPU (parallel_state absent).
                               Use directly for unit tests.
  NeuronGR00TDenoisingWrapper — wraps GR00TDiTModel. Calls nn.Module.__init__ to bypass
                               heavy ModelWrapper init (avoids NxDI distributed setup at
                               construction time). Traceable by torch_neuronx / ModelBuilder.
  NeuronGR00TActionHead       — NeuronActionHeadBase subclass. Owns compile_denoiser() which
                               uses get_builder().trace() with TP=8 parallel_state.

TP=8 rationale:
  Attention Q/K/V: ColumnParallelLinear → splits output heads across 8 NeuronCores.
  Attention output / FFN down: RowParallelLinear → all-reduces partial sums.
  On trn1.32xlarge (32 NeuronCores), TP=8 gives ~6-8x speedup on GEMM ops vs TP=1.
  Each denoising step runs 4x, so per-step latency is critical.

Deviation from AlternateVLDiT:
  All even-indexed blocks attend to ALL conditioning tokens (no image/non-image
  mask alternation). Dynamic image_mask changes per input so can't be a static buffer.
  Documented in notes.md.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
SKILLS_ROOT = os.path.join(os.path.dirname(ROOT), 'groot-port', 'skills', 'scripts')
sys.path.insert(0, ROOT)
sys.path.insert(0, SKILLS_ROOT)
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

from config_constants import (
    BATCH_SIZE, TP_DEGREE_DIT,
    DIT_NUM_LAYERS, DIT_HIDDEN_SIZE, DIT_NUM_HEADS, DIT_HEAD_DIM,
    DIT_INTERMEDIATE_SIZE, DIT_OUTPUT_DIM, DIT_CROSS_ATTN_DIM,
    DIT_INPUT_SEQ_LEN, NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE,
    TIMESTEP_EMBED_DIM,
    MODEL_PATH,
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
# Sub-modules — key naming matches diffusers / HF checkpoint exactly
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    """Adaptive layer norm. Checkpoint keys: norm1.linear.{weight,bias}.
    norm1.norm has elementwise_affine=False → no saved parameters.
    linear kept as nn.Linear (maps temb scalar → scale+shift, not on sequence path).
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        t = self.linear(self.silu(temb))
        scale, shift = t.chunk(2, dim=1)
        return self.norm(x) * (1.0 + scale[:, None]) + shift[:, None]


class _GELU(nn.Module):
    """Matches diffusers GELU(dim_in, dim_out, approximate='tanh').
    Checkpoint keys: net.0.proj.{weight,bias}.
    Uses ColumnParallelLinear for TP split on the up-projection.
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        # ColumnParallelLinear or nn.Linear depending on parallel_state
        self.proj = _col(dim_in, dim_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    """Matches diffusers FeedForward(activation_fn='gelu-approximate').
    net.0 = _GELU (ColumnParallel), net.1 = Dropout, net.2 = RowParallel.
    Checkpoint keys: ff.net.0.proj.{weight,bias}, ff.net.2.{weight,bias}.
    """
    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.ModuleList([
            _GELU(dim, inner_dim),              # net.0: up-proj (column-parallel)
            nn.Dropout(dropout),                 # net.1
            _row(inner_dim, dim, bias=True),     # net.2: down-proj (row-parallel + all-reduce)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[2](self.net[1](self.net[0](x)))


class Attention(nn.Module):
    """Multi-head attention with TP=8 column/row parallel split.
    Checkpoint keys: attn1.to_{q,k,v}.{weight,bias}, attn1.to_out.0.{weight,bias}.

    TP split:
      to_q/to_k/to_v: ColumnParallelLinear(gather_output=False)
        → each rank holds heads/tp heads locally
      to_out[0]: RowParallelLinear(input_is_parallel=True)
        → all-reduces across ranks after output projection
    """
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        cross_attention_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads    = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        kv_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.to_q = _col(query_dim, self.inner_dim, bias=True)
        self.to_k = _col(kv_dim,    self.inner_dim, bias=True)
        self.to_v = _col(kv_dim,    self.inner_dim, bias=True)
        self.to_out = nn.ModuleList([
            _row(self.inner_dim, query_dim, bias=True),  # to_out.0 (row-parallel)
            nn.Dropout(dropout),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, q_len, _ = hidden_states.shape
        kv = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(kv)
        v = self.to_v(kv)

        kv_len = k.shape[1]
        # local_heads = heads // tp_degree when ColumnParallelLinear is active
        local_inner = q.shape[-1]
        local_heads = local_inner // self.dim_head

        q = q.reshape(B, q_len,  local_heads, self.dim_head).transpose(1, 2)
        k = k.reshape(B, kv_len, local_heads, self.dim_head).transpose(1, 2)
        v = v.reshape(B, kv_len, local_heads, self.dim_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn = attn + (1.0 - attention_mask.float()) * -1e9
        attn = F.softmax(attn.float(), dim=-1).to(hidden_states.dtype)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, q_len, local_inner)
        return self.to_out[1](self.to_out[0](out))  # row-parallel all-reduce happens here


class BasicTransformerBlock(nn.Module):
    """Single DiT block. Matches diffusers BasicTransformerBlock checkpoint structure.
    norm1 = AdaLayerNorm (ada_norm), norm3 elementwise_affine=False (no saved weights).
    cross_attention_dim=None → self-attention (odd indices).
    cross_attention_dim set  → cross-attention (even indices).
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int = None,
        dropout: float = 0.0,
        final_dropout: bool = True,
    ):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            dropout=dropout,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None
        self.norm3 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        self.ff = FeedForward(dim=dim, inner_dim=DIT_INTERMEDIATE_SIZE, dropout=dropout)
        self._is_cross = (cross_attention_dim is not None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        norm_hs = self.norm1(hidden_states, temb)
        if self._is_cross:
            attn_out = self.attn1(norm_hs, encoder_hidden_states, encoder_attention_mask)
        else:
            attn_out = self.attn1(norm_hs)
        if self.final_dropout is not None:
            attn_out = self.final_dropout(attn_out)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# GR00TDiTModel — testable standalone nn.Module (no NxDI wrapper overhead)
# ---------------------------------------------------------------------------

class GR00TDiTModel(nn.Module):
    """AlternateVLDiT as a plain nn.Module. Use this for unit tests.

    When parallel_state is not initialized (CPU / unit tests), all
    ColumnParallelLinear / RowParallelLinear fall back to nn.Linear automatically.
    """

    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=DIT_HIDDEN_SIZE,
                num_attention_heads=DIT_NUM_HEADS,
                attention_head_dim=DIT_HEAD_DIM,
                cross_attention_dim=(None if i % 2 == 1 else DIT_CROSS_ATTN_DIM),
                dropout=0.0,
                final_dropout=True,
            )
            for i in range(DIT_NUM_LAYERS)
        ])
        # Output: AdaLN-like scale/shift from temb, then project to action space
        self.norm_out  = nn.LayerNorm(DIT_HIDDEN_SIZE, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(DIT_HIDDEN_SIZE, 2 * DIT_HIDDEN_SIZE)
        self.proj_out_2 = nn.Linear(DIT_HIDDEN_SIZE, DIT_OUTPUT_DIM)

    def forward(
        self,
        sa_embs: torch.Tensor,             # [B, 41, 1536]
        conditioning_tokens: torch.Tensor, # [B, 256, 2048]
        timestep_emb: torch.Tensor,        # [B, 1536]
        cross_attn_mask: torch.Tensor,     # [B, 1, 41, 256] int32
    ) -> torch.Tensor:
        hidden = sa_embs.contiguous()
        cond   = conditioning_tokens.contiguous()
        temb   = timestep_emb

        for i, block in enumerate(self.transformer_blocks):
            if i % 2 == 1:
                hidden = block(hidden, temb)
            else:
                hidden = block(hidden, temb,
                               encoder_hidden_states=cond,
                               encoder_attention_mask=cross_attn_mask)

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden = self.norm_out(hidden) * (1.0 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden)  # [B, 41, 1024]


# ---------------------------------------------------------------------------
# NeuronGR00TDenoisingWrapper — wraps GR00TDiTModel for Neuron compilation
# ---------------------------------------------------------------------------

try:
    from neuron_action_head_base import NeuronDenoisingWrapper as _NeuronDenoisingWrapper
    from neuron_action_head_base import NeuronActionHeadBase, NeuronDenoisingConfig
    _NEURON_BASE_AVAILABLE = True
except ImportError:
    _NeuronDenoisingWrapper = nn.Module
    NeuronActionHeadBase = nn.Module
    NeuronDenoisingConfig = object
    _NEURON_BASE_AVAILABLE = False


class NeuronGR00TDenoisingWrapper(_NeuronDenoisingWrapper):
    """Wraps GR00TDiTModel for torch_neuronx compilation.

    Calls nn.Module.__init__ directly to bypass ModelWrapper's heavy NxDI
    initialization (distributed state setup, memory reservation for all TP ranks).
    This is safe because:
      - torch_neuronx.trace() only needs a valid nn.Module with a forward()
      - NeuronActionHeadBase.get_builder() passes the wrapper to ModelBuilder,
        which initializes parallel_state independently before tracing
      - Unit tests run on CPU where ModelWrapper init would fail / waste RAM

    Pattern: same as SmolVLA vision/backbone wrappers in smol-port-skilled.

    CRITICAL FIX: model is NOT constructed in __init__. It is constructed in
    load_module() AFTER ModelBuilder has initialized parallel_state(tp_degree=8).
    This ensures ColumnParallelLinear/RowParallelLinear use real TP, not fallback.
    """

    def __init__(self):
        nn.Module.__init__(self)   # bypass ModelWrapper.__init__
        self.model = None          # lazy — constructed in load_module()
        self._preload_sd = None    # set by _build_denoising_wrapper before trace

    def forward(
        self,
        sa_embs: torch.Tensor,
        conditioning_tokens: torch.Tensor,
        timestep_emb: torch.Tensor,
        cross_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Auto-init model for pre_compile_validate() on CPU.
        # On Trainium, ModelBuilder calls load_module() explicitly before tracing.
        if self.model is None:
            self.load_module()
        return self.model(sa_embs, conditioning_tokens, timestep_emb, cross_attn_mask)

    def input_generator(self):
        B = BATCH_SIZE
        return [(
            torch.zeros(B, DIT_INPUT_SEQ_LEN,      DIT_HIDDEN_SIZE,          dtype=torch.bfloat16),
            torch.zeros(B, NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE, dtype=torch.bfloat16),
            torch.zeros(B, TIMESTEP_EMBED_DIM,                                dtype=torch.bfloat16),
            torch.ones( B, 1, DIT_INPUT_SEQ_LEN, NUM_CONDITIONING_TOKENS,    dtype=torch.int32),
        )]

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # If model is not yet constructed (pre-load_module), stash weights
        if self.model is None:
            self._preload_sd = state_dict
            return nn.Module.load_state_dict.__func__(nn.Module, self, {}, strict=False)
        return nn.Module.load_state_dict(self, state_dict, strict=strict, **kwargs)

    def load_module(self):
        """ModelBuilder calls this AFTER parallel_state(tp_degree=8) is active.
        Construct GR00TDiTModel here so ColumnParallelLinear uses real TP=8.

        On CPU (pre_compile_validate): loads weights for correctness check.
        On Trainium (ModelBuilder): does NOT load weights (sharding handles this).
        """
        self.model = GR00TDiTModel()
        on_trainium = _parallel_initialized()
        if self._preload_sd is not None and not on_trainium:
            # _preload_sd has 'model.*' prefixed keys (wrapper-level).
            # GR00TDiTModel expects keys without 'model.' prefix.
            model_pfx = 'model.'
            stripped_sd = {}
            for k, v in self._preload_sd.items():
                if k.startswith(model_pfx):
                    stripped_sd[k[len(model_pfx):]] = v
                else:
                    stripped_sd[k] = v
            self.model.load_state_dict(stripped_sd, strict=True)
        self.model = self.model.bfloat16().eval()

    def get(self, bucket_rank=0, **kwargs):
        return self, {}


# ---------------------------------------------------------------------------
# NeuronGR00TActionHead — lifecycle owner, uses ModelBuilder for TP=8 compile
# ---------------------------------------------------------------------------

class NeuronGR00TActionHead(NeuronActionHeadBase):
    """Owns compile/load/warmup lifecycle for the GR00T DiT action head.

    compile_denoiser(save_path, state_dict):
      1. Builds NeuronGR00TDenoisingWrapper
      2. Loads HF weights BEFORE trace (weights baked into NEFF at trace time)
      3. Calls get_builder().trace() — this initializes parallel_state with TP=8
         so ColumnParallelLinear/RowParallelLinear use real TP instead of fallback
    """

    def __init__(self, model_path: str, config: NeuronDenoisingConfig, state_dict: dict = None):
        if _NEURON_BASE_AVAILABLE:
            super().__init__(model_path, config)
        else:
            nn.Module.__init__(self)
            self.model_path = model_path
            self.config = config
        self._preload_sd = state_dict

    def _build_denoising_wrapper(self) -> NeuronGR00TDenoisingWrapper:
        wrapper = NeuronGR00TDenoisingWrapper()
        # Store state dict so load_module() can load it after parallel_state is init
        wrapper._preload_sd = self._preload_sd
        return wrapper

    def _build_attention_mask(self) -> torch.Tensor:
        B = self.config.neuron_config.batch_size
        return torch.ones(B, 1, DIT_INPUT_SEQ_LEN, NUM_CONDITIONING_TOKENS, dtype=torch.int32)

    def checkpoint_loader_fn(self, mmap: bool = False) -> dict:
        return self._preload_sd if self._preload_sd else {}

    def get_compiler_args(self) -> str:
        return "-O1"


# ---------------------------------------------------------------------------
# Weight mapping: HF checkpoint → NeuronGR00TDenoisingWrapper state_dict
# ---------------------------------------------------------------------------

def get_hf_to_neuron_key_mapping() -> dict:
    """Map 'action_head.model.*' checkpoint keys to 'model.*' wrapper keys.

    Layers live under self.model (GR00TDiTModel) in the wrapper.
    Module names match diffusers exactly → strip 'action_head.model.' prefix,
    add 'model.' prefix.
    norm3 has elementwise_affine=False → no saved weights, excluded.
    """
    mapping = {}
    pfx = "action_head.model."

    for i in range(DIT_NUM_LAYERS):
        for sub in (
            "attn1.to_q.weight", "attn1.to_q.bias",
            "attn1.to_k.weight", "attn1.to_k.bias",
            "attn1.to_v.weight", "attn1.to_v.bias",
            "attn1.to_out.0.weight", "attn1.to_out.0.bias",
            "norm1.linear.weight", "norm1.linear.bias",
            "ff.net.0.proj.weight", "ff.net.0.proj.bias",
            "ff.net.2.weight", "ff.net.2.bias",
        ):
            mapping[f"{pfx}transformer_blocks.{i}.{sub}"] = \
                f"model.transformer_blocks.{i}.{sub}"

    for sub in ("proj_out_1.weight", "proj_out_1.bias",
                "proj_out_2.weight", "proj_out_2.bias"):
        mapping[f"{pfx}{sub}"] = f"model.{sub}"

    return mapping


def load_dit_state_dict(hf_sd: dict) -> dict:
    """Extract DiT weights from full GR00T checkpoint for NeuronGR00TDenoisingWrapper.
    Keys are prefixed with 'model.' since layers live under self.model in the wrapper.
    """
    mapping = get_hf_to_neuron_key_mapping()
    out, missing = {}, []
    for hf_key, neuron_key in mapping.items():
        if hf_key in hf_sd:
            out[neuron_key] = hf_sd[hf_key]
        else:
            missing.append(hf_key)
    if missing:
        raise KeyError(f"Missing {len(missing)} checkpoint keys, e.g. {missing[:3]}")
    return out


def load_dit_model_state_dict(hf_sd: dict) -> dict:
    """Extract DiT weights for GR00TDiTModel directly (no 'model.' prefix).
    Use this when loading into GR00TDiTModel for unit tests.
    """
    wrapper_sd = load_dit_state_dict(hf_sd)
    return {k.removeprefix("model."): v for k, v in wrapper_sd.items()}
