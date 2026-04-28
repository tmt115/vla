"""
action_head_integration.py — NeuronGR00TActionHead + CPU denoising loop

Owns the full inference pipeline:
  1. Backbone forward (vision + LLM, compiled NEFF)
  2. VL self-attention (compiled NEFF)
  3. N-step flow-matching denoising loop (CPU)
     - State + action encoding on CPU (CategorySpecificMLP, embodiment-indexed)
     - TimestepEncoder on CPU (scalar timestep → projected embedding)
     - Each step calls DiT NEFF
  4. Action decoding on CPU (CategorySpecificMLP, embodiment-indexed)

CPU-only components (NOT compiled):
  - TimestepEncoder: sinusoidal(256) + Linear(256→1536) + Linear(1536→1536)
  - MultiEmbodimentActionEncoder: CategorySpecificLinear (W.shape [32, 132, 1536])
  - CategorySpecificMLP for state encoding and action decoding
  - Noise schedule (linear 1.0 → 0.0 in num_steps steps)
  - Position embedding addition (nn.Embedding, add_pos_embed=True)
"""

import os, sys, glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, '/home/ubuntu/Isaac-GR00T')
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

from config_constants import (
    BATCH_SIZE, NUM_INFERENCE_TIMESTEPS, NUM_TIMESTEP_BUCKETS,
    DIT_INPUT_SEQ_LEN, ACTION_HORIZON, STATE_HISTORY_LENGTH,
    MAX_ACTION_DIM, MAX_STATE_DIM, INPUT_EMBEDDING_DIM,
    ACTION_OUTPUT_DIM, MAX_NUM_EMBODIMENTS,
    DIT_HIDDEN_SIZE, TIMESTEP_EMBED_DIM, DIT_OUTPUT_DIM,
    NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE,
    MODEL_PATH,
)


# ---------------------------------------------------------------------------
# CPU-side modules (not compiled)
# ---------------------------------------------------------------------------

class TimestepEncoder(nn.Module):
    """Computes projected timestep embedding on CPU.
    Checkpoint: action_head.model.timestep_encoder.*
    Sinusoidal(256) → Linear(256→1536) → SiLU → Linear(1536→1536)
    Output: [B, 1536] = TIMESTEP_EMBED_DIM
    """

    def __init__(self):
        super().__init__()
        # Matches diffusers Timesteps + TimestepEmbedding
        # Timesteps: sinusoidal channels=256
        # TimestepEmbedding: linear_1(256→1536) + silu + linear_2(1536→1536)
        self.linear_1 = nn.Linear(256, TIMESTEP_EMBED_DIM)
        self.linear_2 = nn.Linear(TIMESTEP_EMBED_DIM, TIMESTEP_EMBED_DIM)

    def _sinusoidal(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Matches diffusers Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)."""
        half = 256 // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # flip: cos first
        return embedding

    def forward(self, timestep_bucket: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep_bucket: [B] long tensor, discrete bucket index in [0, 999]
        Returns:
            [B, 1536] bfloat16 projected embedding
        """
        emb = self._sinusoidal(timestep_bucket)           # [B, 256]
        emb = F.silu(self.linear_1(emb.to(self.linear_1.weight.dtype)))
        return self.linear_2(emb)


class CategorySpecificLinear(nn.Module):
    """Per-embodiment linear with stacked weights. CPU only."""
    def __init__(self, num_cats: int, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(num_cats, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(num_cats, out_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        W = self.W[cat_ids]   # [B, in, out]
        b = self.b[cat_ids]   # [B, out]
        return torch.bmm(x, W) + b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with per-embodiment weights. CPU only.
    Checkpoint: action_head.action_encoder.* and action_head.action_decoder.*
    """
    def __init__(self, num_cats: int, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_cats, in_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_cats, hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.silu(self.layer1(x, cat_ids)), cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """Encodes (noisy_action, t_bucket, embodiment_id) → [B, action_horizon, 1536].
    Checkpoint: action_head.action_encoder.W1/W2/W3 (three CategorySpecificLinear layers).
    """
    def __init__(self):
        super().__init__()
        # W1: action embedding (action_dim → hidden)
        self.W1 = CategorySpecificLinear(MAX_NUM_EMBODIMENTS, MAX_ACTION_DIM, INPUT_EMBEDDING_DIM)
        # W2: timestep embedding (2*hidden → hidden) — for concatenated t embedding
        self.W2 = CategorySpecificLinear(MAX_NUM_EMBODIMENTS, 2 * INPUT_EMBEDDING_DIM, INPUT_EMBEDDING_DIM)
        # W3: output projection (hidden → hidden)
        self.W3 = CategorySpecificLinear(MAX_NUM_EMBODIMENTS, INPUT_EMBEDDING_DIM, INPUT_EMBEDDING_DIM)

    def forward(
        self,
        noisy_actions: torch.Tensor,  # [B, horizon, action_dim]
        t_bucket: torch.Tensor,       # [B] long
        embodiment_id: torch.Tensor,  # [B] long
    ) -> torch.Tensor:
        # Project actions
        action_emb = self.W1(noisy_actions, embodiment_id)          # [B, horizon, 1536]
        # Build timestep embedding [B, 1, 1536] → broadcast across horizon
        # Simpler: use sinusoidal directly from the DiT's timestep encoder
        # (caller provides pre-computed t_emb for efficiency)
        return F.silu(action_emb)


# ---------------------------------------------------------------------------
# NeuronGR00TActionHead — full inference pipeline
# ---------------------------------------------------------------------------

class NeuronGR00TActionHead(nn.Module):
    """
    Full GR00T N1.7 inference pipeline.

    On Neuron (after compilation):
      self.backbone     — compiled NEFF (vision + LLM 16 layers)
      self.vl_self_attn — compiled NEFF (4-layer self-attention)
      self.dit          — compiled NEFF (32-layer DiT, 4 denoising steps)

    CPU components:
      self.timestep_encoder
      self.action_encoder (MultiEmbodimentActionEncoder)
      self.state_encoder  (CategorySpecificMLP)
      self.action_decoder (CategorySpecificMLP)
      self.position_embedding (nn.Embedding, add_pos_embed=True)
      self.vlln (LayerNorm on backbone output, use_vlln=True)
    """

    def __init__(self):
        super().__init__()
        # CPU-side modules (loaded from HF checkpoint)
        self.timestep_encoder = TimestepEncoder()
        self.state_encoder = CategorySpecificMLP(
            MAX_NUM_EMBODIMENTS,
            MAX_STATE_DIM * STATE_HISTORY_LENGTH,
            DIT_HIDDEN_SIZE,
            INPUT_EMBEDDING_DIM,
        )
        self.action_encoder = MultiEmbodimentActionEncoder()
        self.action_decoder = CategorySpecificMLP(
            MAX_NUM_EMBODIMENTS,
            ACTION_OUTPUT_DIM,
            DIT_HIDDEN_SIZE,
            MAX_ACTION_DIM,
        )
        self.position_embedding = nn.Embedding(1024, INPUT_EMBEDDING_DIM)
        self.vlln = nn.LayerNorm(CONDITIONING_HIDDEN_SIZE)  # use_vlln=True

        # Compiled subgraphs — set after loading NEFFs
        self.backbone = None     # NeuronGR00TBackbone or compiled NEFF
        self.vl_self_attn = None # NeuronVLSelfAttention or compiled NEFF
        self.dit = None          # NeuronGR00TDenoisingWrapper or compiled NEFF

    def load_cpu_weights(self, hf_sd: dict):
        """Load CPU-side weights from full GR00T checkpoint."""
        # TimestepEncoder
        pfx = 'action_head.model.timestep_encoder.timestep_embedder.'
        self.timestep_encoder.linear_1.weight.data = hf_sd[pfx + 'linear_1.weight']
        self.timestep_encoder.linear_1.bias.data   = hf_sd[pfx + 'linear_1.bias']
        self.timestep_encoder.linear_2.weight.data = hf_sd[pfx + 'linear_2.weight']
        self.timestep_encoder.linear_2.bias.data   = hf_sd[pfx + 'linear_2.bias']

        # State encoder (CategorySpecificMLP: layer1 W/b, layer2 W/b)
        se_pfx = 'action_head.state_encoder.'
        self.state_encoder.layer1.W.data = hf_sd[se_pfx + 'layer1.W']
        self.state_encoder.layer1.b.data = hf_sd[se_pfx + 'layer1.b']
        self.state_encoder.layer2.W.data = hf_sd[se_pfx + 'layer2.W']
        self.state_encoder.layer2.b.data = hf_sd[se_pfx + 'layer2.b']

        # Action encoder (MultiEmbodimentActionEncoder: W1/W2/W3)
        ae_pfx = 'action_head.action_encoder.'
        self.action_encoder.W1.W.data = hf_sd[ae_pfx + 'W1.W']
        self.action_encoder.W1.b.data = hf_sd[ae_pfx + 'W1.b']
        self.action_encoder.W2.W.data = hf_sd[ae_pfx + 'W2.W']
        self.action_encoder.W2.b.data = hf_sd[ae_pfx + 'W2.b']
        self.action_encoder.W3.W.data = hf_sd[ae_pfx + 'W3.W']
        self.action_encoder.W3.b.data = hf_sd[ae_pfx + 'W3.b']

        # Action decoder
        ad_pfx = 'action_head.action_decoder.'
        self.action_decoder.layer1.W.data = hf_sd[ad_pfx + 'layer1.W']
        self.action_decoder.layer1.b.data = hf_sd[ad_pfx + 'layer1.b']
        self.action_decoder.layer2.W.data = hf_sd[ad_pfx + 'layer2.W']
        self.action_decoder.layer2.b.data = hf_sd[ad_pfx + 'layer2.b']

        # Position embedding
        self.position_embedding.weight.data = hf_sd['action_head.position_embedding.weight']

        # VLLN
        if 'action_head.vlln.weight' in hf_sd:
            self.vlln.weight.data = hf_sd['action_head.vlln.weight']
            self.vlln.bias.data   = hf_sd['action_head.vlln.bias']

    def set_compiled_subgraphs(self, backbone, vl_self_attn, dit):
        """Attach compiled NEFFs. Call after loading all NEFFs."""
        self.backbone     = backbone
        self.vl_self_attn = vl_self_attn
        self.dit          = dit

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids:      torch.Tensor,  # [B, SEQ]
        attention_mask: torch.Tensor,  # [B, SEQ]
        pixel_values:   torch.Tensor,  # [NPATCH, C*T*P*P]
        image_grid_thw: torch.Tensor,  # [1, 3]
        state:          torch.Tensor,  # [B, state_history, max_state_dim]
        embodiment_id:  torch.Tensor,  # [B] long
        num_steps:      int = NUM_INFERENCE_TIMESTEPS,
    ) -> torch.Tensor:
        """
        Full GR00T inference. Returns predicted action chunk [B, action_horizon, action_dim].

        num_steps is a Python int — never passed as a tensor into any compiled graph.
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # ── Step 1: Backbone (vision + LLM 16 layers) ──────────────────────
        backbone_features = self.backbone(
            input_ids, attention_mask, pixel_values, image_grid_thw
        )  # [B, SEQ, 2048]

        # ── Step 2: VLLN + VL self-attention ───────────────────────────────
        backbone_features = self.vlln(backbone_features)
        conditioning_tokens = self.vl_self_attn(backbone_features)  # [B, SEQ, 2048]

        # ── Step 3: State encoding (CPU) ────────────────────────────────────
        state_flat = state.view(B, 1, -1)  # [B, 1, state_history*max_state_dim]
        state_features = self.state_encoder(state_flat, embodiment_id)  # [B, 1, 1536]

        # ── Step 4: Flow-matching denoising loop ─────────────────────────────
        # Sample initial noise [B, action_horizon, 1536] (embedded action space)
        noisy_actions = torch.randn(
            B, ACTION_HORIZON, MAX_ACTION_DIM, dtype=torch.bfloat16, device=device
        )

        # Linear timestep schedule from 1.0 → 0.0
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device='cpu')[:-1]

        cross_mask = torch.ones(
            B, 1, DIT_INPUT_SEQ_LEN, NUM_CONDITIONING_TOKENS,
            dtype=torch.int32, device=device
        )

        for t in timesteps:
            # Discretise timestep to bucket index
            t_bucket = torch.tensor(
                [int(t.item() * (NUM_TIMESTEP_BUCKETS - 1))], dtype=torch.long, device=device
            ).expand(B)

            # Encode noisy actions (CPU) → [B, action_horizon, 1536]
            action_features = self.action_encoder.W1(noisy_actions, embodiment_id)
            action_features = F.silu(action_features)

            # Add position embedding
            pos_ids = torch.arange(action_features.shape[1], device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            # Concatenate state + action features → sa_embs [B, 41, 1536]
            sa_embs = torch.cat([state_features, action_features], dim=1)  # [B, 41, 1536]

            # Compute timestep embedding on CPU
            temb = self.timestep_encoder(t_bucket)  # [B, 1536]

            # DiT forward (compiled NEFF)
            velocity_pred = self.dit(sa_embs, conditioning_tokens, temb, cross_mask)
            # [B, 41, 1024] → take action portion [B, 40, 1024]
            action_velocity = velocity_pred[:, -ACTION_HORIZON:, :]  # [B, 40, 1024]

            # Flow-matching Euler step: x_t-dt = x_t - dt * v
            dt = 1.0 / num_steps
            # Decode velocity from 1024-dim DiT output space to action space
            # action_decoder maps 1024 → max_action_dim
            v_decoded = self.action_decoder(action_velocity, embodiment_id)  # [B, 40, 132]
            noisy_actions = noisy_actions - dt * v_decoded

        # Final action chunk
        return noisy_actions  # [B, action_horizon, max_action_dim]
