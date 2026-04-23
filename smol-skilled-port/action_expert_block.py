"""
action_expert_block.py

NeuronSmolVLADenoisingWrapper and NeuronSmolVLAActionHead for AWS Trainium NxDI port.

Architecture:
- 16 expert transformer layers (720-dim hidden)
- Even layers (0,2,...14): self-attention over action tokens
- Odd layers (1,3,...15): cross-attention, Q from action tokens, K/V from VLM KV cache
- GQA: 15 Q heads, 5 KV heads, 64 head dim
- Conditioning packed as [B, 113, 5120] = [B, 113, 8*2*320]
"""

import sys
import math
import os

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

# Base classes — import, do NOT redefine
sys.path.insert(0, '/home/ubuntu/smol-port-skilled/skills/scripts')
from neuron_action_head_base import (
    NeuronDenoisingWrapper,
    NeuronActionHeadBase,
    ConditioningContract,
)

sys.path.insert(0, '/home/ubuntu/smol-port-skilled')
from config_constants import (
    EXPERT_HIDDEN_SIZE,
    EXPERT_NUM_HEADS,
    EXPERT_NUM_KV_HEADS,
    EXPERT_HEAD_DIM,
    EXPERT_INTERMEDIATE_SIZE,
    EXPERT_NUM_LAYERS,
    EXPERT_SELF_ATTN_EVERY_N_LAYERS,
    EXPERT_NUM_CROSS_ATTN_LAYERS,
    EXPERT_CROSS_ATTN_KV_IN,
    EXPERT_CROSS_ATTN_KV_OUT,
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    NUM_CONDITIONING_TOKENS,
    CONDITIONING_HIDDEN_SIZE,
    TIMESTEP_EMBED_DIM,
    MIN_PERIOD,
    MAX_PERIOD,
    NUM_DENOISING_STEPS,
    BATCH_SIZE,
    VLM_KV_DIM,
)


# ---------------------------------------------------------------------------
# Minimal config object for NeuronActionHeadBase.generate_actions()
# ---------------------------------------------------------------------------

class SmolVLAActionHeadConfig:
    """
    Minimal config supplying the attributes that NeuronActionHeadBase.generate_actions()
    needs from self.config. Avoids any NxDI InferenceConfig dependency at test time.
    """
    def __init__(self, batch_size: int = BATCH_SIZE):
        self.action_chunk_size = ACTION_CHUNK_SIZE
        self.action_dim = ACTION_DIM
        self.num_conditioning_tokens = NUM_CONDITIONING_TOKENS
        self.conditioning_hidden_size = CONDITIONING_HIDDEN_SIZE
        self.timestep_embed_dim = TIMESTEP_EMBED_DIM
        self.batch_size = batch_size
        # Needed by NeuronDenoisingWrapper.input_generator() (unused in CPU mode)
        self.neuron_config = type('NeuronConfig', (), {'batch_size': batch_size})()


# ---------------------------------------------------------------------------
# RoPE utility (verbatim from smolvlm_with_expert.py, max_wavelength=10_000)
# ---------------------------------------------------------------------------

def apply_rope_tables(x, sin_table, cos_table):
    """
    Apply RoPE using pre-computed sin/cos tables (registered buffers).
    x:         [B, L, H, D]
    sin_table: [1, L, 1, D//2] float32
    cos_table: [1, L, 1, D//2] float32
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
# GQA attention helper
# ---------------------------------------------------------------------------

def gqa_attention(
    q: torch.Tensor,               # [B, Lq, num_heads, head_dim]
    k: torch.Tensor,               # [B, Lk, num_kv_heads, head_dim]
    v: torch.Tensor,               # [B, Lk, num_kv_heads, head_dim]
    attention_mask: torch.Tensor,  # [B, 1, Lq, Lk] int32 (1=attend, 0=mask)
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:                  # [B, Lq, num_heads * head_dim]
    """
    Grouped-query attention matching eager_attention_forward from SmolVLMWithExpertModel.
    Q/K upcasted to float32, scale = head_dim^-0.5, mask applied additively.
    """
    B = q.shape[0]
    Lq = q.shape[1]
    Lk = k.shape[1]
    num_kv_groups = num_heads // num_kv_heads

    # GQA: expand KV heads from num_kv_heads to num_heads
    # [B, Lk, num_kv_heads, head_dim] -> [B, Lk, num_heads, head_dim]
    k = k[:, :, :, None, :].expand(B, Lk, num_kv_heads, num_kv_groups, head_dim)
    k = k.reshape(B, Lk, num_heads, head_dim)
    v = v[:, :, :, None, :].expand(B, Lk, num_kv_heads, num_kv_groups, head_dim)
    v = v.reshape(B, Lk, num_heads, head_dim)

    # Upcast Q, K to float32 for matmul
    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)

    # Transpose: [B, num_heads, L, head_dim]
    q_t = q_f32.transpose(1, 2)   # [B, num_heads, Lq, head_dim]
    k_t = k_f32.transpose(1, 2)   # [B, num_heads, Lk, head_dim]
    v_t = v.transpose(1, 2)       # [B, num_heads, Lk, head_dim]

    # Compute attention weights
    att_weights = torch.matmul(q_t, k_t.transpose(2, 3))  # [B, num_heads, Lq, Lk]
    att_weights = att_weights * (head_dim ** -0.5)
    att_weights = att_weights.to(torch.float32)

    # Apply mask: int32 1=attend, 0=mask
    big_neg = torch.finfo(torch.float32).min
    mask_bool = attention_mask.bool()  # [B, 1, Lq, Lk]
    att_weights = torch.where(mask_bool, att_weights, big_neg)

    # Softmax + attend
    probs = F.softmax(att_weights, dim=-1)
    probs = probs.to(v.dtype)

    att_output = torch.matmul(probs, v_t)  # [B, num_heads, Lq, head_dim]

    # Reshape: [B, Lq, num_heads * head_dim]
    att_output = att_output.permute(0, 2, 1, 3)
    att_output = att_output.reshape(B, Lq, num_heads * head_dim)

    return att_output


# ---------------------------------------------------------------------------
# Expert layer modules
# ---------------------------------------------------------------------------

class ExpertMLP(nn.Module):
    """SwiGLU MLP for expert transformer layers."""
    def __init__(self):
        super().__init__()
        if _is_parallel():
            g = _tp_group()
            self.gate_proj = ColumnParallelLinear(EXPERT_HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE, bias=False, gather_output=False, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.up_proj   = ColumnParallelLinear(EXPERT_HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE, bias=False, gather_output=False, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.down_proj = RowParallelLinear(EXPERT_INTERMEDIATE_SIZE, EXPERT_HIDDEN_SIZE,    bias=False, input_is_parallel=True,  dtype=torch.bfloat16, tensor_model_parallel_group=g)
        else:
            self.gate_proj = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE, bias=False)
            self.up_proj   = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE, bias=False)
            self.down_proj = nn.Linear(EXPERT_INTERMEDIATE_SIZE, EXPERT_HIDDEN_SIZE, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertSelfAttnLayer(nn.Module):
    """Even-indexed expert layer: self-attention over action tokens (input_dim = EXPERT_HIDDEN_SIZE = 720)."""
    def __init__(self):
        super().__init__()
        if _is_parallel():
            g = _tp_group()
            self.q_proj = ColumnParallelLinear(EXPERT_HIDDEN_SIZE, EXPERT_NUM_HEADS * EXPERT_HEAD_DIM,    bias=False, gather_output=True, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.k_proj = ColumnParallelLinear(EXPERT_HIDDEN_SIZE, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False, gather_output=True, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.v_proj = ColumnParallelLinear(EXPERT_HIDDEN_SIZE, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False, gather_output=True, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.o_proj = RowParallelLinear(EXPERT_NUM_HEADS * EXPERT_HEAD_DIM, EXPERT_HIDDEN_SIZE,        bias=False, input_is_parallel=False, dtype=torch.bfloat16, tensor_model_parallel_group=g)
        else:
            self.q_proj = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_NUM_HEADS * EXPERT_HEAD_DIM,    bias=False)
            self.k_proj = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False)
            self.v_proj = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False)
            self.o_proj = nn.Linear(EXPERT_NUM_HEADS * EXPERT_HEAD_DIM, EXPERT_HIDDEN_SIZE,     bias=False)
        self.input_layernorm = nn.RMSNorm(EXPERT_HIDDEN_SIZE, eps=1e-5)
        self.post_attention_layernorm = nn.RMSNorm(EXPERT_HIDDEN_SIZE, eps=1e-5)
        self.mlp = ExpertMLP()
        self.head_dim = EXPERT_HEAD_DIM


class ExpertCrossAttnLayer(nn.Module):
    """Odd-indexed expert layer: cross-attention, Q from action tokens, K/V from VLM (input_dim = 320)."""
    def __init__(self):
        super().__init__()
        if _is_parallel():
            g = _tp_group()
            self.q_proj = ColumnParallelLinear(EXPERT_HIDDEN_SIZE,       EXPERT_NUM_HEADS * EXPERT_HEAD_DIM,    bias=False, gather_output=True, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.k_proj = ColumnParallelLinear(EXPERT_CROSS_ATTN_KV_IN,  EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False, gather_output=True, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.v_proj = ColumnParallelLinear(EXPERT_CROSS_ATTN_KV_IN,  EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False, gather_output=True, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.o_proj = RowParallelLinear(EXPERT_NUM_HEADS * EXPERT_HEAD_DIM, EXPERT_HIDDEN_SIZE,              bias=False, input_is_parallel=False, dtype=torch.bfloat16, tensor_model_parallel_group=g)
        else:
            self.q_proj = nn.Linear(EXPERT_HIDDEN_SIZE,      EXPERT_NUM_HEADS * EXPERT_HEAD_DIM,    bias=False)
            self.k_proj = nn.Linear(EXPERT_CROSS_ATTN_KV_IN, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False)
            self.v_proj = nn.Linear(EXPERT_CROSS_ATTN_KV_IN, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False)
            self.o_proj = nn.Linear(EXPERT_NUM_HEADS * EXPERT_HEAD_DIM, EXPERT_HIDDEN_SIZE,          bias=False)
        self.input_layernorm = nn.RMSNorm(EXPERT_HIDDEN_SIZE, eps=1e-5)
        self.post_attention_layernorm = nn.RMSNorm(EXPERT_HIDDEN_SIZE, eps=1e-5)
        self.mlp = ExpertMLP()
        self.head_dim = EXPERT_HEAD_DIM


# ---------------------------------------------------------------------------
# NeuronSmolVLADenoisingWrapper
# ---------------------------------------------------------------------------

class NeuronSmolVLADenoisingWrapper(NeuronDenoisingWrapper):
    """
    Single compiled denoising step for SmolVLA action expert on Trainium.

    Inputs (all static shapes for XLA tracing):
        noisy_actions:       [B, 50, 32]    BF16
        conditioning_tokens: [B, 113, 5120] BF16  (packed VLM KV: 8 layers × 2 (K,V) × 320)
        timestep_embedding:  [B, 720]       BF16  (sinusoidal, computed on CPU)
        attention_mask:      [B, 1, 50, 113] INT32

    Output: v_t [B, 50, 32] BF16  (velocity prediction for flow matching)
    """

    def __init__(self, config=None):
        # NeuronDenoisingWrapper.super().__init__() calls ModelWrapper which requires
        # config+model_cls positional args — bypass with nn.Module.__init__ directly.
        nn.Module.__init__(self)
        self.config = config

        if _is_parallel():
            g = _tp_group()
            self.action_in_proj      = ColumnParallelLinear(ACTION_DIM, EXPERT_HIDDEN_SIZE,            bias=True, gather_output=True,     dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.action_time_mlp_in  = ColumnParallelLinear(EXPERT_HIDDEN_SIZE * 2, EXPERT_HIDDEN_SIZE, bias=True, gather_output=True,     dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.action_time_mlp_out = ColumnParallelLinear(EXPERT_HIDDEN_SIZE, EXPERT_HIDDEN_SIZE,     bias=True, gather_output=True,     dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.action_out_proj     = RowParallelLinear(EXPERT_HIDDEN_SIZE, ACTION_DIM,               bias=True, input_is_parallel=False, dtype=torch.bfloat16, tensor_model_parallel_group=g)
        else:
            self.action_in_proj      = nn.Linear(ACTION_DIM,            EXPERT_HIDDEN_SIZE)
            self.action_time_mlp_in  = nn.Linear(EXPERT_HIDDEN_SIZE * 2, EXPERT_HIDDEN_SIZE)
            self.action_time_mlp_out = nn.Linear(EXPERT_HIDDEN_SIZE,    EXPERT_HIDDEN_SIZE)
            self.action_out_proj     = nn.Linear(EXPERT_HIDDEN_SIZE,    ACTION_DIM)

        # 16 interleaved expert layers
        self.layers = nn.ModuleList()
        for i in range(EXPERT_NUM_LAYERS):
            if i % EXPERT_SELF_ATTN_EVERY_N_LAYERS == 0:
                self.layers.append(ExpertSelfAttnLayer())
            else:
                self.layers.append(ExpertCrossAttnLayer())

        # Final RMSNorm
        self.lm_expert_norm = nn.RMSNorm(EXPERT_HIDDEN_SIZE, eps=1e-5)

        # Pre-computed RoPE tables for action self-attention (L=ACTION_CHUNK_SIZE).
        # Registered as buffers so they appear as constants in the traced graph —
        # avoids torch.arange/timescale being re-created on every denoising step.
        _d_half = EXPERT_HEAD_DIM // 2
        _positions = torch.arange(ACTION_CHUNK_SIZE, dtype=torch.float32)
        _freq_exp  = (2.0 / EXPERT_HEAD_DIM) * torch.arange(_d_half, dtype=torch.float32)
        _timescale = 10_000.0 ** _freq_exp
        _radians   = _positions[:, None] / _timescale[None, :]       # [L, d_half]
        self.register_buffer('rope_sin', torch.sin(_radians)[None, :, None, :])  # [1, L, 1, d_half]
        self.register_buffer('rope_cos', torch.cos(_radians)[None, :, None, :])

        # Static self-attention mask (all action tokens attend to all action tokens).
        self.register_buffer(
            'self_attn_mask',
            torch.ones(BATCH_SIZE, 1, ACTION_CHUNK_SIZE, ACTION_CHUNK_SIZE, dtype=torch.int32),
        )

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # ModelWrapper overrides load_state_dict and calls self.model_cls which we don't set.
        # Pass **kwargs (e.g. assign=True from torch_neuronx internals) through to nn.Module.
        return nn.Module.load_state_dict(self, state_dict, strict=strict, **kwargs)

    def input_generator(self):
        # Override base class which reads from self.config.neuron_config — config is None
        # here since we bypass NeuronDenoisingWrapper.__init__. Use constants directly.
        return [(
            torch.zeros(BATCH_SIZE, ACTION_CHUNK_SIZE, ACTION_DIM, dtype=torch.bfloat16),
            torch.zeros(BATCH_SIZE, NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE,
                        dtype=torch.bfloat16),
            torch.zeros(BATCH_SIZE, TIMESTEP_EMBED_DIM, dtype=torch.bfloat16),
            torch.zeros(BATCH_SIZE, 1, ACTION_CHUNK_SIZE, NUM_CONDITIONING_TOKENS,
                        dtype=torch.int32),
        )]

    def compile(self, save_path: str) -> None:
        """
        Compile the denoising wrapper to a NEFF.

        Trace() is called here, inside the ModelWrapper subclass, per NxDI convention.
        Called from NeuronActionHeadBase.compile_denoiser() — do not call torch_neuronx.trace()
        outside this method.
        """
        import torch_neuronx
        from config_constants import EXPERT_FLAGS
        example_inputs = self.input_generator()[0]
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        traced = torch_neuronx.trace(
            self,
            example_inputs,
            compiler_args=EXPERT_FLAGS,
            compiler_workdir=os.path.join(os.path.dirname(save_path), 'compiler_workdir_expert'),
        )
        traced.save(save_path)
        # Replace self with the compiled ScriptModule so generate_actions() calls the NEFF
        return traced

    def forward(
        self,
        noisy_actions: torch.Tensor,       # [B, 50, 32] BF16
        conditioning_tokens: torch.Tensor, # [B, 113, 5120] BF16
        timestep_embedding: torch.Tensor,  # [B, 720] BF16
        attention_mask: torch.Tensor,      # [B, 1, 50, 113] INT32
    ) -> torch.Tensor:                     # [B, 50, 32] BF16
        B = noisy_actions.shape[0]
        wdtype = self.action_in_proj.weight.dtype
        noisy_actions       = noisy_actions.to(wdtype)
        conditioning_tokens = conditioning_tokens.to(wdtype)
        timestep_embedding  = timestep_embedding.to(wdtype)

        # ── Step 1–5: embed action tokens ──────────────────────────────────
        # 1. Linear projection: [B, 50, 32] → [B, 50, 720]
        action_emb = self.action_in_proj(noisy_actions)

        # 2. Expand sinusoidal timestep: [B, 720] → [B, 50, 720]
        time_emb = timestep_embedding[:, None, :].expand(B, ACTION_CHUNK_SIZE, EXPERT_HIDDEN_SIZE)

        # 3. Concatenate: [B, 50, 1440]
        fused = torch.cat([action_emb, time_emb], dim=2)

        # 4. SiLU MLP: [B, 50, 1440] → [B, 50, 720]
        fused = F.silu(self.action_time_mlp_in(fused))

        # 5. Final MLP: [B, 50, 720]
        hidden = self.action_time_mlp_out(fused)

        # ── Step 6: unpack conditioning tokens ─────────────────────────────
        # [B, 113, 5120] → [B, 113, 8, 2, 320]
        # Dimension layout: 8 cross-attn layers × (K=0, V=1) × 320 VLM KV dim
        conditioning_reshaped = conditioning_tokens.reshape(
            B, NUM_CONDITIONING_TOKENS, EXPERT_NUM_CROSS_ATTN_LAYERS, 2, VLM_KV_DIM
        )

        # Pre-computed buffers (registered in __init__) — no dynamic tensor creation.
        self_attn_mask = self.self_attn_mask

        # ── Step 7: run 16 interleaved expert layers ─────────────────────────
        for layer_idx in range(EXPERT_NUM_LAYERS):
            layer = self.layers[layer_idx]

            if layer_idx % EXPERT_SELF_ATTN_EVERY_N_LAYERS == 0:
                # ─── EVEN LAYER: self-attention over action tokens ──────────
                normed = layer.input_layernorm(hidden)

                # Q: [B, 50, 720] → [B, 50, 960] → [B, 50, 15, 64]
                q = layer.q_proj(normed).reshape(B, ACTION_CHUNK_SIZE, EXPERT_NUM_HEADS, EXPERT_HEAD_DIM)
                # K: [B, 50, 720] → [B, 50, 320] → [B, 50, 5, 64]
                k = layer.k_proj(normed).reshape(B, ACTION_CHUNK_SIZE, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)
                # V: [B, 50, 720] → [B, 50, 320] → [B, 50, 5, 64]
                v = layer.v_proj(normed).reshape(B, ACTION_CHUNK_SIZE, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)

                # RoPE on Q and K using pre-computed tables
                q = apply_rope_tables(q, self.rope_sin, self.rope_cos)
                k = apply_rope_tables(k, self.rope_sin, self.rope_cos)

                att_output = gqa_attention(
                    q, k, v, self_attn_mask,
                    EXPERT_NUM_HEADS, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM
                )

            else:
                # ─── ODD LAYER: cross-attention with VLM KV cache ──────────
                # Cross-attn layer index: layers 1,3,5,...15 → c = 0,1,2,...7
                c = layer_idx // 2

                # Extract K, V from packed conditioning
                k_vlm = conditioning_reshaped[:, :, c, 0, :]  # [B, 113, 320]
                v_vlm = conditioning_reshaped[:, :, c, 1, :]  # [B, 113, 320]

                normed = layer.input_layernorm(hidden)

                # Q: [B, 50, 720] → [B, 50, 960] → [B, 50, 15, 64]
                q = layer.q_proj(normed).reshape(B, ACTION_CHUNK_SIZE, EXPERT_NUM_HEADS, EXPERT_HEAD_DIM)

                # K: VLM KV [B, 113, 320] → expert K_proj [320→320] → [B, 113, 5, 64]
                k = layer.k_proj(k_vlm).reshape(B, NUM_CONDITIONING_TOKENS, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)
                # V: VLM KV [B, 113, 320] → expert V_proj [320→320] → [B, 113, 5, 64]
                v = layer.v_proj(v_vlm).reshape(B, NUM_CONDITIONING_TOKENS, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)

                # RoPE on Q only (K is NOT RoPE'd in cross-attn, per HF reference)
                q = apply_rope_tables(q, self.rope_sin, self.rope_cos)

                att_output = gqa_attention(
                    q, k, v, attention_mask,  # [B, 1, 50, 113]
                    EXPERT_NUM_HEADS, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM
                )

            # ─── O projection + first residual ─────────────────────────────
            out = layer.o_proj(att_output.to(layer.o_proj.weight.dtype))
            hidden = hidden + out

            # ─── Post-attention norm + MLP + second residual ───────────────
            after_attn = hidden.clone()
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = hidden + after_attn

        # ── Step 8–9: final norm + output projection ─────────────────────────
        hidden = self.lm_expert_norm(hidden)
        v_t = self.action_out_proj(hidden)

        return v_t


# ---------------------------------------------------------------------------
# NeuronSmolVLAActionHead
# ---------------------------------------------------------------------------

class NeuronSmolVLAActionHead(NeuronActionHeadBase):
    """
    SmolVLA action head on AWS Trainium NxDI.

    Manages:
        - compile_denoiser(): traces NeuronSmolVLADenoisingWrapper to a NEFF
        - generate_actions(): runs the 10-step flow-matching denoising loop
    """

    def __init__(self, config=None):
        # NeuronActionHeadBase.super() calls NeuronApplicationBase which requires model_path.
        # Bypass with nn.Module.__init__ and set required attrs manually.
        nn.Module.__init__(self)
        if config is None:
            config = SmolVLAActionHeadConfig()
        self.config = config
        self.denoising_wrapper = None

    def get_conditioning_contract(self) -> ConditioningContract:
        return ConditioningContract(
            num_conditioning_tokens=NUM_CONDITIONING_TOKENS,
            conditioning_hidden_size=CONDITIONING_HIDDEN_SIZE,
        )

    def _build_denoising_wrapper(self) -> NeuronSmolVLADenoisingWrapper:
        return NeuronSmolVLADenoisingWrapper(self.config)

    def _get_timestep_sequence(self, num_steps: int):
        """Flow-matching timestep schedule: 1.0, 0.9, 0.8, ..., 0.1"""
        return [1.0 - i / num_steps for i in range(num_steps)]

    def _embed_timestep(self, t: float) -> torch.Tensor:
        """Compute sinusoidal timestep embedding on CPU → [B, 720] BF16."""
        from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
        cpu = torch.device("cpu")
        t_tensor = torch.tensor([t], dtype=torch.float32, device=cpu)
        emb = create_sinusoidal_pos_embedding(t_tensor, TIMESTEP_EMBED_DIM, MIN_PERIOD, MAX_PERIOD, device=cpu)
        return emb.to(torch.bfloat16).expand(BATCH_SIZE, -1)  # [B, 720]

    def _build_attention_mask(self) -> torch.Tensor:
        """All-ones: every action token attends to every conditioning token."""
        return torch.ones(
            BATCH_SIZE, 1, ACTION_CHUNK_SIZE, NUM_CONDITIONING_TOKENS,
            dtype=torch.int32,
        )

    def compile_denoiser(self, save_path: str) -> None:
        """
        Build and compile the denoising wrapper to a NEFF.

        Delegates compilation to NeuronSmolVLADenoisingWrapper.compile() — trace() is
        called inside the ModelWrapper subclass, not here. On CPU (no Neuron hardware),
        the wrapper is built but compile() is a no-op and the plain nn.Module is used.
        """
        self.denoising_wrapper = self._build_denoising_wrapper()
        self.attention_mask = self._build_attention_mask()
        if hasattr(self.denoising_wrapper, 'compile'):
            traced = self.denoising_wrapper.compile(save_path)
            if traced is not None:
                self.denoising_wrapper = traced


# ---------------------------------------------------------------------------
# PyTorch Reference: direct architectural reimplementation for weight-synced testing
# ---------------------------------------------------------------------------

class SmolVLADenoisingStepRef(nn.Module):
    """
    PyTorch reference implementation of the SmolVLA denoising step.

    This is a pure-PyTorch faithful replica of NeuronSmolVLADenoisingWrapper,
    used for weight-synchronized correctness testing.

    Weights should be loaded from the installed lerobot model (no new weights created here).
    The weight mapping from HF checkpoint keys to this module's keys is given by
    get_weight_mapping().
    """

    def __init__(self):
        super().__init__()
        # Identical architecture to NeuronSmolVLADenoisingWrapper
        self.action_in_proj = nn.Linear(ACTION_DIM, EXPERT_HIDDEN_SIZE)
        self.action_time_mlp_in = nn.Linear(EXPERT_HIDDEN_SIZE * 2, EXPERT_HIDDEN_SIZE)
        self.action_time_mlp_out = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_HIDDEN_SIZE)
        self.action_out_proj = nn.Linear(EXPERT_HIDDEN_SIZE, ACTION_DIM)

        self.layers = nn.ModuleList()
        for i in range(EXPERT_NUM_LAYERS):
            if i % EXPERT_SELF_ATTN_EVERY_N_LAYERS == 0:
                self.layers.append(ExpertSelfAttnLayer())
            else:
                self.layers.append(ExpertCrossAttnLayer())

        self.lm_expert_norm = nn.RMSNorm(EXPERT_HIDDEN_SIZE, eps=1e-5)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        conditioning_tokens: torch.Tensor,
        timestep_embedding: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Identical logic to NeuronSmolVLADenoisingWrapper.forward()."""
        B = noisy_actions.shape[0]

        action_emb = self.action_in_proj(noisy_actions)
        time_emb = timestep_embedding[:, None, :].expand(B, ACTION_CHUNK_SIZE, EXPERT_HIDDEN_SIZE)
        fused = torch.cat([action_emb, time_emb], dim=2)
        fused = F.silu(self.action_time_mlp_in(fused))
        hidden = self.action_time_mlp_out(fused)

        conditioning_reshaped = conditioning_tokens.reshape(
            B, NUM_CONDITIONING_TOKENS, EXPERT_NUM_CROSS_ATTN_LAYERS, 2, VLM_KV_DIM
        )

        action_pos_ids = torch.arange(
            ACTION_CHUNK_SIZE, device=noisy_actions.device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)

        self_attn_mask = torch.ones(
            B, 1, ACTION_CHUNK_SIZE, ACTION_CHUNK_SIZE,
            dtype=torch.int32, device=noisy_actions.device
        )

        for layer_idx in range(EXPERT_NUM_LAYERS):
            layer = self.layers[layer_idx]

            if layer_idx % EXPERT_SELF_ATTN_EVERY_N_LAYERS == 0:
                normed = layer.input_layernorm(hidden)
                q = layer.q_proj(normed).reshape(B, ACTION_CHUNK_SIZE, EXPERT_NUM_HEADS, EXPERT_HEAD_DIM)
                k = layer.k_proj(normed).reshape(B, ACTION_CHUNK_SIZE, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)
                v = layer.v_proj(normed).reshape(B, ACTION_CHUNK_SIZE, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)
                q = apply_rope(q, action_pos_ids)
                k = apply_rope(k, action_pos_ids)
                att_output = gqa_attention(q, k, v, self_attn_mask,
                                           EXPERT_NUM_HEADS, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)
            else:
                c = layer_idx // 2
                k_vlm = conditioning_reshaped[:, :, c, 0, :]
                v_vlm = conditioning_reshaped[:, :, c, 1, :]
                normed = layer.input_layernorm(hidden)
                q = layer.q_proj(normed).reshape(B, ACTION_CHUNK_SIZE, EXPERT_NUM_HEADS, EXPERT_HEAD_DIM)
                k = layer.k_proj(k_vlm).reshape(B, NUM_CONDITIONING_TOKENS, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)
                v = layer.v_proj(v_vlm).reshape(B, NUM_CONDITIONING_TOKENS, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)
                q = apply_rope(q, action_pos_ids)
                att_output = gqa_attention(q, k, v, attention_mask,
                                           EXPERT_NUM_HEADS, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)

            out = layer.o_proj(att_output.to(layer.o_proj.weight.dtype))
            hidden = hidden + out

            after_attn = hidden.clone()
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = hidden + after_attn

        hidden = self.lm_expert_norm(hidden)
        return self.action_out_proj(hidden)


# ---------------------------------------------------------------------------
# Weight mapping: SmolVLADenoisingStepRef → NeuronSmolVLADenoisingWrapper
# ---------------------------------------------------------------------------

def get_weight_mapping():
    """
    Returns a dict mapping PyTorch reference state dict keys to Neuron wrapper keys.
    Both SmolVLADenoisingStepRef and NeuronSmolVLADenoisingWrapper share the same
    architecture and naming, so the mapping is 1:1.
    """
    mapping = {}

    # Top-level projections
    for key in ("action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"):
        for suffix in ("weight", "bias"):
            full_key = f"{key}.{suffix}"
            mapping[full_key] = full_key

    # Final norm
    mapping["lm_expert_norm.weight"] = "lm_expert_norm.weight"

    # Expert layers
    for layer_idx in range(EXPERT_NUM_LAYERS):
        pfx = f"layers.{layer_idx}"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            k = f"{pfx}.{proj}.weight"
            mapping[k] = k
        for norm in ("input_layernorm", "post_attention_layernorm"):
            k = f"{pfx}.{norm}.weight"
            mapping[k] = k
        for mlp in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            k = f"{pfx}.{mlp}.weight"
            mapping[k] = k

    return mapping


def get_hf_to_neuron_weight_mapping():
    """
    Returns a dict mapping HuggingFace checkpoint keys (model.safetensors)
    to NeuronSmolVLADenoisingWrapper state dict keys.

    HF prefix: model.vlm_with_expert.lm_expert.layers.N.self_attn.*
    Neuron prefix: layers.N.*
    """
    mapping = {}

    # Action projections: model.action_* → action_*
    for key in ("action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"):
        for suffix in ("weight", "bias"):
            mapping[f"model.{key}.{suffix}"] = f"{key}.{suffix}"

    # Final norm: model.vlm_with_expert.lm_expert.norm.weight → lm_expert_norm.weight
    mapping["model.vlm_with_expert.lm_expert.norm.weight"] = "lm_expert_norm.weight"

    # Expert layers
    for layer_idx in range(EXPERT_NUM_LAYERS):
        src_pfx = f"model.vlm_with_expert.lm_expert.layers.{layer_idx}"
        dst_pfx = f"layers.{layer_idx}"

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mapping[f"{src_pfx}.self_attn.{proj}.weight"] = f"{dst_pfx}.{proj}.weight"

        for norm in ("input_layernorm", "post_attention_layernorm"):
            mapping[f"{src_pfx}.{norm}.weight"] = f"{dst_pfx}.{norm}.weight"

        for mlp in ("gate_proj", "up_proj", "down_proj"):
            mapping[f"{src_pfx}.mlp.{mlp}.weight"] = f"{dst_pfx}.mlp.{mlp}.weight"

    return mapping


def load_from_checkpoint(model: NeuronSmolVLADenoisingWrapper, checkpoint_path: str) -> None:
    """
    Load weights from the SmolVLA safetensors checkpoint into the Neuron wrapper.
    """
    from safetensors.torch import load_file

    sd = load_file(checkpoint_path)
    hf_mapping = get_hf_to_neuron_weight_mapping()

    new_sd = {}
    for hf_key, neuron_key in hf_mapping.items():
        if hf_key in sd:
            new_sd[neuron_key] = sd[hf_key]
        else:
            print(f"  WARNING: HF key not found: {hf_key}")

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"Loaded {len(new_sd)} weights from checkpoint.")
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
