"""
suffix_denoiser.py — π0 Gemma 300M suffix denoiser for Trainium.

One step of the flow-matching denoising loop. Takes noisy actions + state +
timestep embedding + prefix KV cache → velocity prediction v_t.

Compiled graph inputs (all STATIC shapes):
    state:         [1, 32]                     bfloat16
    noisy_actions: [1, 50, 32]                 bfloat16
    time_emb:      [1, 1024]                   bfloat16  (CPU-computed sinusoidal)
    prefix_kv:     [18, 2, 1, 816, 1, 256]    bfloat16  (from prefix encoder)

Output:
    v_t:           [1, 50, 32]                 float32   (velocity for Euler step)
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gemma.modeling_gemma import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

sys.path.insert(0, '/home/ubuntu/pi0-port/skills/scripts')
from neuron_action_head_base import (
    NeuronDenoisingWrapper,
    NeuronDenoisingConfig,
    NeuronActionHeadBase,
    ConditioningContract,
)

from config_constants import (
    EXPERT_HIDDEN_SIZE, EXPERT_NUM_LAYERS, EXPERT_NUM_HEADS,
    EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM, EXPERT_INTERMEDIATE_SIZE,
    EXPERT_RMS_NORM_EPS, CHUNK_SIZE, MAX_STATE_DIM, MAX_ACTION_DIM,
    PREFIX_LEN, SUFFIX_LEN, FULL_ATTN_LEN, MIN_PERIOD, MAX_PERIOD,
)


# ── Attention mask construction ──────────────────────────────────────────────

def _build_suffix_attn_mask(prefix_len: int = PREFIX_LEN,
                             chunk_size: int = CHUNK_SIZE,
                             large_neg: float = -1e9) -> torch.Tensor:
    """
    Build [1, 1, suffix_len, full_len] additive attention mask.

    Suffix att_masks = [1, 1, 0×(chunk_size-1)]:
      cumsum = [1, 2, 2, ..., 2]
    Prefix cumsum = 0 for all positions.

    Suffix position i (cumsum=ci) attends to position j if cumsum[j] <= ci:
      State (ci=1):  prefix (0<=1) ✓, state (1<=1) ✓, actions (2<=1) ✗
      Actions (ci=2): prefix (0<=2) ✓, all suffix (2<=2) ✓
    """
    suffix_len = 1 + chunk_size
    full_len = prefix_len + suffix_len
    mask = torch.zeros(1, 1, suffix_len, full_len, dtype=torch.float32)
    # State token (row 0): block action positions [prefix_len+1 .. full_len-1]
    mask[0, 0, 0, prefix_len + 1:] = large_neg
    # Action tokens (rows 1..chunk_size): attend everywhere — already 0.0
    return mask


# ── NeuronPi0DenoisingWrapper ─────────────────────────────────────────────────

class NeuronPi0DenoisingWrapper(NeuronDenoisingWrapper):
    """
    Single-step π0 suffix denoiser with custom forward signature.

    Manually iterates the Gemma 300M layers so we can inject the prefix KV
    at each layer's attention — avoiding DynamicCache (not XLA-traceable).
    """

    def __init__(self, config: NeuronDenoisingConfig,
                 num_layers: int = EXPERT_NUM_LAYERS,
                 hidden_size: int = EXPERT_HIDDEN_SIZE,
                 head_dim: int = EXPERT_HEAD_DIM,
                 num_heads: int = EXPERT_NUM_HEADS,
                 kv_heads: int = EXPERT_NUM_KV_HEADS,
                 intermediate_size: int = EXPERT_INTERMEDIATE_SIZE,
                 rms_norm_eps: float = EXPERT_RMS_NORM_EPS,
                 prefix_len: int = PREFIX_LEN,
                 chunk_size: int = CHUNK_SIZE,
                 max_state: int = MAX_STATE_DIM,
                 max_action: int = MAX_ACTION_DIM):
        # CRITICAL: bypass ModelWrapper.__init__ — LLM-oriented, allocates ~350 GB RAM
        nn.Module.__init__(self)
        self.config = config
        self.model = None             # constructed in load_module()
        self._preload_sd = None

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.prefix_len = prefix_len
        self.chunk_size = chunk_size
        self.max_state = max_state
        self.max_action = max_action
        self.suffix_len = 1 + chunk_size

        # Action projection layers (populated in load_module)
        self.state_proj = None
        self.action_in_proj = None
        self.action_time_mlp_in = None
        self.action_time_mlp_out = None
        self.action_out_proj = None

        # Static buffers
        mask = _build_suffix_attn_mask(prefix_len, chunk_size)
        self.register_buffer("suffix_attn_mask", mask)                 # [1,1,51,867]
        pos = torch.arange(prefix_len, prefix_len + self.suffix_len).unsqueeze(0)
        self.register_buffer("suffix_position_ids", pos)               # [1,51]

    def load_module(self):
        """Called by ModelBuilder AFTER parallel_state is initialized."""
        from transformers.models.gemma.modeling_gemma import GemmaConfig
        from lerobot.policies.pi_gemma import PiGemmaForCausalLM

        cfg = GemmaConfig(
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_heads,
            num_hidden_layers=self.num_layers,
            num_key_value_heads=self.kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            rms_norm_eps=self.rms_norm_eps,
            use_adarms=False,
            adarms_cond_dim=None,
        )
        cfg._attn_implementation = "eager"

        expert = PiGemmaForCausalLM(config=cfg)
        expert.model.embed_tokens = None  # suffix uses pre-computed embeddings

        self.state_proj = nn.Linear(self.max_state, self.hidden_size)
        self.action_in_proj = nn.Linear(self.max_action, self.hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.action_time_mlp_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_out_proj = nn.Linear(self.hidden_size, self.max_action)

        self.model = expert

        if self._preload_sd is not None:
            self.load_state_dict(self._preload_sd, strict=False)

        # Cast attention/MLP to bfloat16; layernorms stay float32
        self.model = self.model.bfloat16()
        for proj in [self.state_proj, self.action_in_proj,
                     self.action_time_mlp_in, self.action_time_mlp_out,
                     self.action_out_proj]:
            proj.to(torch.bfloat16)
        for name, param in self.named_parameters():
            if "layernorm" in name or name.endswith(".norm.weight"):
                param.data = param.data.to(torch.float32)

        self.eval()

    def _embed_suffix(self, state, noisy_actions, time_emb):
        """Embed state + noisy actions + timestep into suffix_embs [B, 51, H]."""
        w = self.state_proj.weight.dtype
        state_emb = self.state_proj(state.to(w)).unsqueeze(1)          # [B, 1, H]
        action_emb = self.action_in_proj(noisy_actions.to(w))          # [B, 50, H]
        time_exp = time_emb.to(w).unsqueeze(1).expand(-1, self.chunk_size, -1)  # [B, 50, H]
        action_time = torch.cat([action_emb, time_exp], dim=2)         # [B, 50, 2H]
        action_time = F.silu(self.action_time_mlp_in(action_time))
        action_time = self.action_time_mlp_out(action_time)            # [B, 50, H]
        return torch.cat([state_emb, action_time], dim=1)              # [B, 51, H]

    def forward(
        self,
        state: torch.Tensor,          # [B, 32] bfloat16
        noisy_actions: torch.Tensor,  # [B, 50, 32] bfloat16
        time_emb: torch.Tensor,       # [B, H] bfloat16
        prefix_kv: torch.Tensor,      # [L, 2, B, prefix_len, 1, head_dim] bfloat16
    ) -> torch.Tensor:                # [B, 50, 32] float32

        suffix_embs = self._embed_suffix(state, noisy_actions, time_emb)  # [B, 51, H]
        B, suffix_len, H = suffix_embs.shape

        # Compute rotary embeddings for suffix positions
        expert_layers = self.model.model.layers
        rotary_emb = self.model.model.rotary_emb
        cos, sin = rotary_emb(suffix_embs, self.suffix_position_ids)

        hidden_states = suffix_embs

        attn_mask_bf16 = self.suffix_attn_mask.to(hidden_states.dtype)

        for i, layer in enumerate(expert_layers):
            attn = layer.self_attn
            hd  = attn.head_dim
            nh  = attn.q_proj.weight.shape[0] // hd
            nkv = attn.k_proj.weight.shape[0] // hd

            # input layernorm
            normed, _ = layer.input_layernorm(hidden_states, cond=None)
            w = normed.dtype

            # Q, K, V for suffix tokens
            q = attn.q_proj(normed).view(B, suffix_len, nh,  hd).transpose(1, 2)   # [B, nh, 51, hd]
            k = attn.k_proj(normed).view(B, suffix_len, nkv, hd).transpose(1, 2)   # [B, 1, 51, hd]
            v = attn.v_proj(normed).view(B, suffix_len, nkv, hd).transpose(1, 2)   # [B, 1, 51, hd]

            # Apply RoPE to Q and K (suffix positions only)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Retrieve prefix K, V from the input tensor
            # prefix_kv[i]: [2, B, prefix_len, 1, hd]
            # Layout: [k=0/v=1, B, seq, kv_heads, head_dim] → need [B, kv_heads, seq, head_dim]
            pk = prefix_kv[i, 0].permute(0, 2, 1, 3)   # [B, 1, prefix_len, hd]
            pv = prefix_kv[i, 1].permute(0, 2, 1, 3)   # [B, 1, prefix_len, hd]

            # Concatenate prefix and suffix KV
            k_full = torch.cat([pk, k], dim=2)   # [B, 1, prefix_len+51, hd]
            v_full = torch.cat([pv, v], dim=2)

            # Attention: Q over suffix [B, nh, 51, hd], K/V over full [B, 1, 867, hd]
            attn_out, _ = eager_attention_forward(
                attn, q, k_full, v_full, attn_mask_bf16, scaling=attn.scaling,
            )
            attn_out = attn_out.reshape(B, suffix_len, nh * hd)
            if attn_out.dtype != attn.o_proj.weight.dtype:
                attn_out = attn_out.to(attn.o_proj.weight.dtype)
            attn_out = attn.o_proj(attn_out)

            # First residual
            hidden_states = hidden_states + attn_out

            # Post-attention layernorm
            normed2, _ = layer.post_attention_layernorm(hidden_states, cond=None)
            if normed2.dtype != layer.mlp.up_proj.weight.dtype:
                normed2 = normed2.to(layer.mlp.up_proj.weight.dtype)

            # MLP + second residual
            hidden_states = hidden_states + layer.mlp(normed2)

        # Final norm (from PiGemmaModel — optional but matches reference)
        final_norm = self.model.model.norm
        suffix_out, _ = final_norm(hidden_states, cond=None)

        # Slice last chunk_size tokens (action tokens) and project
        action_out = suffix_out[:, -self.chunk_size:]                       # [B, 50, H]
        v_t = self.action_out_proj(action_out.to(self.action_out_proj.weight.dtype))
        return v_t.to(torch.float32)

    def input_generator(self):
        B = 1
        return [(
            torch.zeros(B, self.max_state, dtype=torch.bfloat16),
            torch.zeros(B, self.chunk_size, self.max_action, dtype=torch.bfloat16),
            torch.zeros(B, self.hidden_size, dtype=torch.bfloat16),
            torch.zeros(self.num_layers, 2, B, self.prefix_len, self.kv_heads,
                        self.head_dim, dtype=torch.bfloat16),
        )]

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Call nn.Module.load_state_dict directly (bypass ModelWrapper).
        # torch_neuronx internally passes assign=True — **kwargs ensures it is forwarded.
        return nn.Module.load_state_dict(self, state_dict, strict=strict, **kwargs)

    def get(self, bucket_rank: int = 0, **kwargs):
        # ModelBuilder._generate_hlo() calls model_instance.get(bucket_rank) to obtain
        # the callable module and input/output aliases. Newer NxDI removed this from
        # ModelWrapper, so we implement it directly here.
        return self, {}  # (callable_module, input_output_aliases)


# ── NeuronPi0ActionHead ───────────────────────────────────────────────────────

class NeuronPi0ActionHead(NeuronActionHeadBase):
    """Owns the π0 flow-matching denoising loop on Trainium."""

    def __init__(self, model_path: str, batch_size: int = 1):
        config = NeuronDenoisingConfig(
            batch_size=batch_size,
            tp_degree=1,
            action_chunk_size=CHUNK_SIZE,
            action_dim=MAX_ACTION_DIM,
            num_conditioning_tokens=PREFIX_LEN,
            conditioning_hidden_size=EXPERT_HIDDEN_SIZE,
            timestep_embed_dim=EXPERT_HIDDEN_SIZE,
        )
        super().__init__(model_path=model_path, config=config)
        self.batch_size = batch_size

        # Pre-compute sinusoidal period vector (avoids float64 / torch.linspace in forward)
        fraction = torch.linspace(0.0, 1.0, EXPERT_HIDDEN_SIZE // 2, dtype=torch.float32)
        period = MIN_PERIOD * (MAX_PERIOD / MIN_PERIOD) ** fraction
        self.register_buffer("period_vector", period)

    def _build_denoising_wrapper(self) -> NeuronPi0DenoisingWrapper:
        # Build wrapper, call load_module() so layers are initialized for pre_compile_validate().
        # compile_denoiser() calls this before pre_compile_validate(), which needs valid weights.
        wrapper = NeuronPi0DenoisingWrapper(self.config)
        wrapper.load_module()
        # Load checkpoint weights so CPU forward pass in pre_compile_validate succeeds.
        try:
            sd = self.get_state_dict(self.model_path, self.config)
            wrapper.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"[suffix] Warning: could not load weights into wrapper: {e}")
        return wrapper

    def _get_timestep_sequence(self, num_steps: int):
        dt = 1.0 / num_steps
        return [1.0 - i * dt for i in range(num_steps)]

    def _embed_timestep(self, t: float) -> torch.Tensor:
        time_tensor = torch.full((self.batch_size,), t, dtype=torch.float32)
        scaling = 1.0 / self.period_vector * 2 * math.pi
        sin_input = scaling[None, :] * time_tensor[:, None]
        time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
        return time_emb.to(torch.bfloat16)

    def _build_attention_mask(self):
        return None  # wrapper uses its own pre-computed buffer

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        """
        Map π0 checkpoint keys (after `model.` prefix strip) to NeuronPi0DenoisingWrapper keys.

        Checkpoint keys after `model.` strip:
          action_in_proj.*             → action_in_proj.*          (unchanged)
          action_out_proj.*            → action_out_proj.*
          state_proj.*                 → state_proj.*
          action_time_mlp_in.*         → action_time_mlp_in.*
          action_time_mlp_out.*        → action_time_mlp_out.*
          paligemma_with_expert.gemma_expert.model.layers.*
                                       → model.model.layers.*
          paligemma_with_expert.gemma_expert.model.norm.*
                                       → model.model.norm.*
          paligemma_with_expert.gemma_expert.lm_head.*
                                       → skip (not in wrapper)
        """
        out = {}
        expert_prefix = "paligemma_with_expert.gemma_expert.model."
        skip_prefixes = ("paligemma_with_expert.paligemma.", "paligemma_with_expert.gemma_expert.lm_head.")
        for k, v in state_dict.items():
            if any(k.startswith(s) for s in skip_prefixes):
                continue
            if k.startswith(expert_prefix):
                new_k = "model.model." + k[len(expert_prefix):]
                out[new_k] = v
            else:
                out[k] = v  # action_in_proj, action_out_proj, state_proj, etc.
        return out

    def get_conditioning_contract(self) -> ConditioningContract:
        return ConditioningContract(
            num_conditioning_tokens=PREFIX_LEN,
            conditioning_hidden_size=EXPERT_HIDDEN_SIZE,
        )

    def generate_actions_with_kv(
        self,
        prefix_kv: torch.Tensor,
        state: torch.Tensor,
        num_steps: int = 10,
        actual_action_dim: int = 6,
    ) -> torch.Tensor:
        """
        Run π0 flow-matching denoising loop.

        Args:
            prefix_kv:        [18, 2, 1, 816, 1, 256] bfloat16
            state:            [1, state_dim] float32
            num_steps:        number of Euler denoising steps (int)
            actual_action_dim: unpadded action dimension for output slicing
        """
        assert isinstance(num_steps, int)

        B = prefix_kv.shape[2]
        state_padded = state.to(torch.bfloat16)
        if state_padded.shape[-1] < MAX_STATE_DIM:
            state_padded = F.pad(state_padded, (0, MAX_STATE_DIM - state_padded.shape[-1]))

        x_t = torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM, dtype=torch.bfloat16)
        dt = -1.0 / num_steps

        for t in self._get_timestep_sequence(num_steps):
            time_emb = self._embed_timestep(t)
            if self.is_neuron():
                # ModelBuilder NEFF: call traced_model directly (weights already initialized)
                v_t = self.traced_model(state_padded, x_t, time_emb, prefix_kv)
            else:
                # CPU fallback: run through Python forward
                assert self.denoising_wrapper is not None, "Call compile_denoiser() or load() first"
                v_t = self.denoising_wrapper(state_padded, x_t, time_emb, prefix_kv)
            x_t = x_t + dt * v_t.to(torch.bfloat16)

        return x_t[:, :, :actual_action_dim]


def load_suffix_denoiser(checkpoint_path: str) -> NeuronPi0DenoisingWrapper:
    """Load action expert weights from the lerobot/pi0 checkpoint."""
    from safetensors.torch import load_file

    config = NeuronDenoisingConfig(
        batch_size=1, tp_degree=1,
        action_chunk_size=CHUNK_SIZE, action_dim=MAX_ACTION_DIM,
        num_conditioning_tokens=PREFIX_LEN, conditioning_hidden_size=EXPERT_HIDDEN_SIZE,
        timestep_embed_dim=EXPERT_HIDDEN_SIZE,
    )
    wrapper = NeuronPi0DenoisingWrapper(config)
    wrapper.load_module()

    sd = load_file(checkpoint_path)

    # Action projection keys (root-level in checkpoint)
    proj_map = {
        'model.action_in_proj.weight':      'action_in_proj.weight',
        'model.action_in_proj.bias':        'action_in_proj.bias',
        'model.action_out_proj.weight':     'action_out_proj.weight',
        'model.action_out_proj.bias':       'action_out_proj.bias',
        'model.state_proj.weight':          'state_proj.weight',
        'model.state_proj.bias':            'state_proj.bias',
        'model.action_time_mlp_in.weight':  'action_time_mlp_in.weight',
        'model.action_time_mlp_in.bias':    'action_time_mlp_in.bias',
        'model.action_time_mlp_out.weight': 'action_time_mlp_out.weight',
        'model.action_time_mlp_out.bias':   'action_time_mlp_out.bias',
    }

    expert_src_prefix = 'model.paligemma_with_expert.gemma_expert.'
    target_sd = {}

    for src_k, tgt_k in proj_map.items():
        if src_k in sd:
            target_sd[tgt_k] = sd[src_k]

    for k, v in sd.items():
        if k.startswith(expert_src_prefix):
            # model.paligemma_with_expert.gemma_expert.model.layers.0.* → model.model.layers.0.*
            new_k = 'model.' + k[len(expert_src_prefix):]
            target_sd[new_k] = v

    missing, unexpected = wrapper.load_state_dict(target_sd, strict=False)
    skip = {"embed_tokens", "lm_head", "embed_tokens"}
    non_trivial_missing = [k for k in missing if not any(s in k for s in skip)]
    if non_trivial_missing:
        print(f"load_suffix_denoiser: {len(non_trivial_missing)} missing: {non_trivial_missing[:5]}")

    return wrapper
