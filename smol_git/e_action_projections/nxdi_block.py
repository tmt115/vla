"""
NeuronSmolVLAActionProjections

Ports the action/state projection layers from VLAFlowMatching:

  state_proj          Linear(32 → 960)     maps robot state → VLM embedding space
  action_in_proj      Linear(32 → 720)     maps noisy actions → expert embedding space
  action_out_proj     Linear(720 → 32)     maps expert output → action space
  action_time_mlp_in  Linear(1440 → 720)   fuses [action_emb, time_emb]
  action_time_mlp_out Linear(720 → 720)    second MLP layer

Also provides create_sinusoidal_pos_embedding_neuron — a float32-only, XLA-compatible
replacement for lerobot's create_sinusoidal_pos_embedding which calls get_safe_dtype
and tries to use float64 (unsupported on Trainium).

Shape conventions (matching static shape table):
  state:          [B, 32]          float32
  noisy_actions:  [B, S, 32]       float32  (S = chunk_size = 50)
  timestep:       [B]              float32
  suffix_embs:    [B, S, 720]      bfloat16
  state_emb:      [B, 1, 960]      bfloat16
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Try to import NxDI parallel primitives; fall back to plain nn for CPU tests
# ---------------------------------------------------------------------------
try:
    import neuronx_distributed.parallel_layers.parallel_state as parallel_state
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    HAS_NXD = True
except Exception:
    HAS_NXD = False


def _col_linear(in_f: int, out_f: int, bias: bool = False) -> nn.Module:
    if HAS_NXD and parallel_state.model_parallel_is_initialized():
        return ColumnParallelLinear(
            in_f, out_f, bias=bias, gather_output=True,
            dtype=torch.bfloat16,
        )
    return nn.Linear(in_f, out_f, bias=bias)


def _row_linear(in_f: int, out_f: int, bias: bool = False) -> nn.Module:
    if HAS_NXD and parallel_state.model_parallel_is_initialized():
        return RowParallelLinear(
            in_f, out_f, bias=bias, input_is_parallel=False,
            dtype=torch.bfloat16,
        )
    return nn.Linear(in_f, out_f, bias=bias)


# ---------------------------------------------------------------------------
# XLA-compatible sinusoidal embedding
# ---------------------------------------------------------------------------

def create_sinusoidal_pos_embedding_neuron(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    """
    Float32-only, XLA-compatible sinusoidal positional embedding.

    Args:
        time:        [B] float32 — timestep values in [0, 1]
        dimension:   must be even
        min_period:  e.g. 4e-3
        max_period:  e.g. 4.0

    Returns:
        [B, dimension] float32
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    device = time.device
    # float32 throughout — no float64 on XLA
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction          # [dim//2]
    scaling_factor = 1.0 / period * 2.0 * math.pi                       # [dim//2]

    # outer product: [B, dim//2]
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)  # [B, dim]
    return pos_emb


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class NeuronSmolVLAActionProjections(nn.Module):
    """
    Encapsulates ALL projection layers involved in building prefix/suffix
    embeddings for the SmolVLA action-expert pipeline.

    This module is used in two separate compiled subgraphs:
      - Prefix pass: forward_state  (called once, result is part of prefix_embs)
      - Denoise step: forward_suffix (called 10× in the Euler loop)

    Both forward methods are pure-tensor → pure-tensor so they can be traced
    by torch_neuronx.trace independently.

    Dimensions (hardcoded to SmolVLA-base defaults):
      MAX_STATE_DIM    = 32
      MAX_ACTION_DIM   = 32
      EXPERT_HIDDEN    = 720   (= 0.75 × VLM_HIDDEN)
      VLM_HIDDEN       = 960
      MIN_PERIOD       = 4e-3
      MAX_PERIOD       = 4.0
    """

    MAX_STATE_DIM  = 32
    MAX_ACTION_DIM = 32
    EXPERT_HIDDEN  = 720
    VLM_HIDDEN     = 960
    MIN_PERIOD     = 4e-3
    MAX_PERIOD     = 4.0

    def __init__(
        self,
        config=None,           # accepted for API compatibility with test harness
        max_state_dim:  int   = 32,
        max_action_dim: int   = 32,
        expert_hidden:  int   = 720,
        vlm_hidden:     int   = 960,
        min_period:     float = 4e-3,
        max_period:     float = 4.0,
    ):
        super().__init__()
        self.max_state_dim  = max_state_dim
        self.max_action_dim = max_action_dim
        self.expert_hidden  = expert_hidden
        self.vlm_hidden     = vlm_hidden
        self.min_period     = min_period
        self.max_period     = max_period

        # state → VLM embedding space
        self.state_proj = _col_linear(max_state_dim, vlm_hidden, bias=True)

        # action → expert embedding space
        self.action_in_proj  = _col_linear(max_action_dim, expert_hidden, bias=True)

        # expert output → action space
        self.action_out_proj = _col_linear(expert_hidden, max_action_dim, bias=True)

        # fuse [action_emb, time_emb] then project
        self.action_time_mlp_in  = _col_linear(expert_hidden * 2, expert_hidden, bias=True)
        self.action_time_mlp_out = _col_linear(expert_hidden, expert_hidden, bias=True)

        # Precompute sinusoidal scaling factors as a buffer so they are
        # saved in the checkpoint and loaded to the Neuron device.
        # This avoids torch.linspace inside traced forward (problematic on XLA).
        fraction = torch.linspace(0.0, 1.0, expert_hidden // 2, dtype=torch.float32)
        period   = min_period * (max_period / min_period) ** fraction
        scaling  = 1.0 / period * 2.0 * math.pi          # [expert_hidden // 2]
        self.register_buffer("sine_scale", scaling, persistent=True)

    # ------------------------------------------------------------------
    # Sub-forward: prefix state embedding
    # ------------------------------------------------------------------

    def forward_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Map robot state to VLM embedding space.

        Args:
            state: [B, max_state_dim] float32

        Returns:
            state_emb: [B, 1, vlm_hidden] (same dtype as state_proj weights)
        """
        emb = self.state_proj(state)           # [B, vlm_hidden]
        return emb.unsqueeze(1)                # [B, 1, vlm_hidden]

    # ------------------------------------------------------------------
    # Sub-forward: suffix action+time embedding (called per denoise step)
    # ------------------------------------------------------------------

    def forward_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed noisy_actions and timestep into the action-expert token sequence.

        Replicates VLAFlowMatching.embed_suffix (embedding part only).

        Args:
            noisy_actions: [B, S, max_action_dim]   float32 or bfloat16
            timestep:      [B]                       float32

        Returns:
            suffix_embs: [B, S, expert_hidden]  same dtype as noisy_actions
        """
        dtype  = noisy_actions.dtype
        device = noisy_actions.device
        B, S, _ = noisy_actions.shape

        action_emb = self.action_in_proj(noisy_actions)   # [B, S, expert_hidden]

        # sinusoidal time embedding — use precomputed buffer to avoid torch.linspace in XLA trace
        sin_input = self.sine_scale[None, :] * timestep[:, None]           # [B, expert_hidden//2] float32
        time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)  # [B, expert_hidden]
        time_emb = time_emb.to(dtype=dtype)
        time_emb = time_emb[:, None, :].expand(B, S, self.expert_hidden)

        action_time_emb = torch.cat([action_emb, time_emb], dim=2)   # [B, S, 2*expert_hidden]

        action_time_emb = self.action_time_mlp_in(action_time_emb)   # [B, S, expert_hidden]
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)  # [B, S, expert_hidden]

        return action_time_emb

    # ------------------------------------------------------------------
    # Convenience: combined forward used during tracing suffix subgraph
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Default forward = suffix embedding (used by torch_neuronx.trace for
        the denoise-step subgraph).

        For the prefix subgraph, call forward_state directly.
        """
        return self.forward_suffix(noisy_actions, timestep)
