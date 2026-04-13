"""
Unit test for NeuronSmolVLAActionProjections.

Tests two modes:
  1. Suffix embedding  forward(noisy_actions, timestep) → [B, S, 720]
  2. State embedding   forward_state(state) → [B, 1, 960]

The test harness (block_testing_utils.test_block_correctness) calls the
reference block as reference_block(reference_inputs[0][0]), i.e. single-arg.
Both reference wrappers are therefore single-arg; for the suffix test a fixed
timestep is baked into the reference and matched in test_inputs.
"""
import sys
sys.path.insert(0, '/home/ubuntu/smol_port/tests')
import _transformers_compat  # bypass huggingface-hub version check — must be first

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/ubuntu/smol_port/e_action_projections')
from nxdi_block import NeuronSmolVLAActionProjections, create_sinusoidal_pos_embedding_neuron

from block_testing_utils import test_block_correctness

# ---------------------------------------------------------------------------
# Architecture constants (SmolVLA-base defaults)
# ---------------------------------------------------------------------------
MAX_STATE_DIM  = 32
MAX_ACTION_DIM = 32
EXPERT_HIDDEN  = 720
VLM_HIDDEN     = 960
MIN_PERIOD     = 4e-3
MAX_PERIOD     = 4.0
CHUNK_SIZE     = 50   # suffix length
BATCH_SIZE     = 1

# Fixed timestep used for suffix test (chosen to exercise non-trivial sinusoidal values)
FIXED_TIMESTEP = 0.3


# ===========================================================================
# TEST 1: Suffix embedding  (action + timestep → expert tokens)
# ===========================================================================

class SuffixEmbeddingRef(nn.Module):
    """
    Reference: replicates VLAFlowMatching.embed_suffix (embedding portion only).

    Uses float32 sinusoidal embedding (no float64 / get_safe_dtype).
    Timestep is fixed at construction time so forward() is single-arg
    (required by block_testing_utils which calls reference_block(tensor)).
    """
    def __init__(self, timestep_value: float = FIXED_TIMESTEP, batch_size: int = BATCH_SIZE):
        super().__init__()
        self.action_in_proj      = nn.Linear(MAX_ACTION_DIM, EXPERT_HIDDEN, bias=True)
        self.action_time_mlp_in  = nn.Linear(EXPERT_HIDDEN * 2, EXPERT_HIDDEN, bias=True)
        self.action_time_mlp_out = nn.Linear(EXPERT_HIDDEN, EXPERT_HIDDEN, bias=True)
        self.expert_hidden = EXPERT_HIDDEN
        # Plain float — NOT a buffer, so harness .to(bfloat16) won't degrade precision
        self._timestep_val = float(timestep_value)
        self._batch_size   = batch_size

    def forward(self, noisy_actions: torch.Tensor) -> torch.Tensor:
        """
        noisy_actions: [B, S, MAX_ACTION_DIM]
        Returns:       [B, S, EXPERT_HIDDEN]
        """
        dtype  = noisy_actions.dtype
        device = noisy_actions.device
        B, S, _ = noisy_actions.shape

        action_emb = self.action_in_proj(noisy_actions)   # [B, S, 720]

        # Compute sinusoidal embedding in float32 inline (reference is not XLA-traced)
        # This always runs in float32 regardless of the .to(bfloat16) cast by the harness
        frac = torch.linspace(0.0, 1.0, self.expert_hidden // 2, dtype=torch.float32, device=device)
        period    = MIN_PERIOD * (MAX_PERIOD / MIN_PERIOD) ** frac
        sine_scale = 1.0 / period * 2.0 * math.pi                            # [expert_hidden//2]
        timestep  = torch.full((B,), self._timestep_val, dtype=torch.float32, device=device)
        sin_input = sine_scale[None, :] * timestep[:, None]
        time_emb  = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)  # [B, 720]
        time_emb  = time_emb.to(dtype=dtype)
        time_emb  = time_emb[:, None, :].expand(B, S, self.expert_hidden)

        action_time_emb = torch.cat([action_emb, time_emb], dim=2)      # [B, S, 1440]
        action_time_emb = self.action_time_mlp_in(action_time_emb)       # [B, S, 720]
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)      # [B, S, 720]

        return action_time_emb


# Weight mapping: ref keys → neuron block keys (harness auto-prepends "block.")
SUFFIX_WEIGHT_MAP = {
    "action_in_proj.weight":      "action_in_proj.weight",
    "action_in_proj.bias":        "action_in_proj.bias",
    "action_time_mlp_in.weight":  "action_time_mlp_in.weight",
    "action_time_mlp_in.bias":    "action_time_mlp_in.bias",
    "action_time_mlp_out.weight": "action_time_mlp_out.weight",
    "action_time_mlp_out.bias":   "action_time_mlp_out.bias",
    # sine_scale is a deterministic float32 constant — not synced; both blocks compute it the same way
}


def test_suffix_embedding():
    B, S = BATCH_SIZE, CHUNK_SIZE
    torch.manual_seed(2)
    noisy_actions = torch.randn(B, S, MAX_ACTION_DIM, dtype=torch.bfloat16)

    # Fixed timestep must match what SuffixEmbeddingRef uses internally
    fixed_t = torch.full((B,), FIXED_TIMESTEP, dtype=torch.float32)

    example_actions  = torch.zeros(B, S, MAX_ACTION_DIM, dtype=torch.bfloat16)
    example_timestep = torch.full((B,), FIXED_TIMESTEP, dtype=torch.float32)

    test_block_correctness(
        neuron_block_class=NeuronSmolVLAActionProjections,
        pytorch_block_class=SuffixEmbeddingRef,
        weight_mapping=SUFFIX_WEIGHT_MAP,
        neuron_init_kwargs={
            "max_state_dim":  MAX_STATE_DIM,
            "max_action_dim": MAX_ACTION_DIM,
            "expert_hidden":  EXPERT_HIDDEN,
            "vlm_hidden":     VLM_HIDDEN,
            "min_period":     MIN_PERIOD,
            "max_period":     MAX_PERIOD,
        },
        pytorch_init_kwargs={
            "timestep_value": FIXED_TIMESTEP,
            "batch_size":     B,
        },
        example_inputs=[(example_actions, example_timestep)],
        test_inputs=[(noisy_actions, fixed_t)],
        reference_inputs=[(noisy_actions,)],   # harness calls ref(noisy_actions) only
        checkpoint_name="e_action_proj_suffix.pt",
        batch_size=B,
        seq_len=S,
        hidden_size=EXPERT_HIDDEN,
        verbose=True,
    )
    print("PASSED: Action projections suffix embedding")


# ===========================================================================
# TEST 2: State embedding  (state → VLM prefix token)
# ===========================================================================

class StateEmbeddingRef(nn.Module):
    """
    Reference: VLAFlowMatching.embed_prefix state projection.
    state_proj: Linear(32 → 960, bias=True), output unsqueezed to [B, 1, 960].
    """
    def __init__(self):
        super().__init__()
        self.state_proj = nn.Linear(MAX_STATE_DIM, VLM_HIDDEN, bias=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: [B, 32] → [B, 1, 960]"""
        return self.state_proj(state).unsqueeze(1)


class NeuronSmolVLAStateEmbedder(NeuronSmolVLAActionProjections):
    """
    Thin subclass that routes forward() → forward_state() so that
    test_block_correctness can compile and call it as a single-input block.
    """
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward_state(state)


STATE_WEIGHT_MAP = {
    "state_proj.weight": "state_proj.weight",
    "state_proj.bias":   "state_proj.bias",
}


def test_state_embedding():
    B = BATCH_SIZE
    torch.manual_seed(3)
    state = torch.randn(B, MAX_STATE_DIM, dtype=torch.bfloat16)

    example_state = torch.zeros(B, MAX_STATE_DIM, dtype=torch.bfloat16)

    test_block_correctness(
        neuron_block_class=NeuronSmolVLAStateEmbedder,
        pytorch_block_class=StateEmbeddingRef,
        weight_mapping=STATE_WEIGHT_MAP,
        neuron_init_kwargs={
            "max_state_dim":  MAX_STATE_DIM,
            "max_action_dim": MAX_ACTION_DIM,
            "expert_hidden":  EXPERT_HIDDEN,
            "vlm_hidden":     VLM_HIDDEN,
            "min_period":     MIN_PERIOD,
            "max_period":     MAX_PERIOD,
        },
        pytorch_init_kwargs={},
        example_inputs=[(example_state,)],
        test_inputs=[(state,)],
        reference_inputs=[(state,)],
        checkpoint_name="e_action_proj_state.pt",
        batch_size=B,
        seq_len=1,
        hidden_size=VLM_HIDDEN,
        verbose=True,
    )
    print("PASSED: Action projections state embedding")


# ===========================================================================
# Standalone unit: validate sinusoidal embedding float32 accuracy
# ===========================================================================

def test_sinusoidal_embedding_accuracy():
    """
    Verify float32 sinusoidal embedding is close to the float64 reference.
    Max absolute error should be well within bfloat16 precision (~0.01).
    """
    B = 4
    DIM = EXPERT_HIDDEN
    torch.manual_seed(7)
    t = torch.rand(B, dtype=torch.float32)

    emb_f32 = create_sinusoidal_pos_embedding_neuron(t, DIM, MIN_PERIOD, MAX_PERIOD)

    # float64 reference
    fraction  = torch.linspace(0.0, 1.0, DIM // 2, dtype=torch.float64)
    period    = MIN_PERIOD * (MAX_PERIOD / MIN_PERIOD) ** fraction
    scaling   = 1.0 / period * 2.0 * math.pi
    sin_input = scaling[None, :] * t[:, None].double()
    emb_f64   = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).float()

    max_err = (emb_f32 - emb_f64).abs().max().item()
    assert max_err < 2e-4, f"Sinusoidal embedding float32 vs float64 max_err={max_err:.2e}"
    print(f"PASSED: Sinusoidal embedding float32 accuracy (max_err={max_err:.2e})")


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sinusoidal embedding accuracy check")
    print("=" * 60)
    test_sinusoidal_embedding_accuracy()

    print("\n" + "=" * 60)
    print("TEST 1: Suffix embedding (action + timestep)")
    print("=" * 60)
    test_suffix_embedding()

    print("\n" + "=" * 60)
    print("TEST 2: State embedding")
    print("=" * 60)
    test_state_embedding()

    print("\nAll action projection tests PASSED")
