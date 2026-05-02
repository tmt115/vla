"""
run_inference.py — GR00T N1.7 inference on AWS Trainium

Loads all compiled NEFFs and CPU weights, then runs inference.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python run_inference.py

Environment:
    GROOT_MODEL_PATH    — override default checkpoint path
    GROOT_COMPILED_DIR  — override default compiled/ directory
"""

import os
import sys
import glob
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'skills', 'scripts'))
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

from safetensors.torch import load_file

from config_constants import (
    BATCH_SIZE, MODEL_PATH, NUM_INFERENCE_TIMESTEPS, NUM_TIMESTEP_BUCKETS,
    DIT_INPUT_SEQ_LEN, ACTION_HORIZON, STATE_HISTORY_LENGTH,
    MAX_ACTION_DIM, MAX_STATE_DIM, INPUT_EMBEDDING_DIM,
    ACTION_OUTPUT_DIM, MAX_NUM_EMBODIMENTS,
    DIT_HIDDEN_SIZE, TIMESTEP_EMBED_DIM, DIT_OUTPUT_DIM,
    NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE,
    VL_SELF_ATTN_HIDDEN_SIZE, LLM_HIDDEN_SIZE,
    VIT_PATCH_SIZE, VIT_TEMPORAL_PATCH_SIZE, VIT_SPATIAL_MERGE_SIZE,
    VISION_TOKENS_PER_IMAGE,
    TP_DEGREE_VLM, TP_DEGREE_DIT,
)
from vlm_backbone_block import (
    NeuronGR00TBackboneHead, NeuronGR00TBackboneWrapper,
    load_backbone_state_dict, _make_backbone_compile_config,
    run_vit_cpu, make_backbone_inputs,
)
from vl_self_attention_block import (
    NeuronGR00TVLSelfAttnWrapper, load_vl_self_attn_state_dict,
)
from dit_block import (
    NeuronGR00TActionHead, NeuronGR00TDenoisingWrapper,
    load_dit_state_dict,
)
from neuron_action_head_base import NeuronDenoisingConfig

COMPILED_DIR = os.environ.get(
    'GROOT_COMPILED_DIR',
    os.path.join(ROOT, 'compiled'),
)
BACKBONE_DIR = os.path.join(COMPILED_DIR, 'backbone')
VL_SA_DIR    = os.path.join(COMPILED_DIR, 'vl_self_attn')
DIT_DIR      = os.path.join(COMPILED_DIR, 'dit')

MODEL_FILE = 'model.pt'


# ---------------------------------------------------------------------------
# CPU-side modules (not compiled)
# ---------------------------------------------------------------------------

class TimestepEncoder(nn.Module):
    """Sinusoidal timestep encoding + 2-layer MLP. CPU only."""

    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(256, TIMESTEP_EMBED_DIM)
        self.linear_2 = nn.Linear(TIMESTEP_EMBED_DIM, TIMESTEP_EMBED_DIM)

    def _sinusoidal(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = 256 // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps.float()[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, timestep_bucket: torch.Tensor) -> torch.Tensor:
        emb = self._sinusoidal(timestep_bucket)
        emb = F.silu(self.linear_1(emb.to(self.linear_1.weight.dtype)))
        return self.linear_2(emb)


class CategorySpecificLinear(nn.Module):
    """Per-embodiment linear. CPU only."""

    def __init__(self, num_cats: int, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(num_cats, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(num_cats, out_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        W = self.W[cat_ids]
        b = self.b[cat_ids]
        return torch.bmm(x, W) + b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer per-embodiment MLP. CPU only."""

    def __init__(self, num_cats: int, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_cats, in_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_cats, hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.silu(self.layer1(x, cat_ids)), cat_ids)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class GR00TModel:
    """
    Container for all compiled subgraphs + CPU components.
    """

    def __init__(self):
        # CPU modules
        self.timestep_encoder = TimestepEncoder()
        self.state_encoder = None   # CategorySpecificMLP, dims from checkpoint
        self.action_encoder_W1 = None  # CategorySpecificLinear
        self.action_decoder = None  # CategorySpecificMLP, dims from checkpoint
        self.position_embedding = nn.Embedding(1024, INPUT_EMBEDDING_DIM)
        self.vlln = nn.LayerNorm(CONDITIONING_HIDDEN_SIZE)

        # Compiled subgraphs
        self.backbone_wrapper = None    # NeuronGR00TBackboneWrapper
        self.vl_self_attn_wrapper = None  # NeuronGR00TVLSelfAttnWrapper
        self.dit_wrapper = None         # NeuronGR00TDenoisingWrapper

        # VIT model for CPU preprocessing
        self._hf_sd = None

    def _load_hf_sd(self) -> dict:
        if self._hf_sd is None:
            print("Loading checkpoint...")
            files = sorted(glob.glob(os.path.join(MODEL_PATH, '*.safetensors')))
            sd = {}
            for f in files:
                sd.update(load_file(f))
            self._hf_sd = sd
            print(f"  Loaded {len(sd)} tensors")
        return self._hf_sd

    def load_cpu_weights(self, hf_sd: dict) -> None:
        """Load CPU-side weights from full GR00T checkpoint."""
        # TimestepEncoder
        pfx = 'action_head.model.timestep_encoder.timestep_embedder.'
        self.timestep_encoder.linear_1.weight.data = hf_sd[pfx + 'linear_1.weight']
        self.timestep_encoder.linear_1.bias.data   = hf_sd[pfx + 'linear_1.bias']
        self.timestep_encoder.linear_2.weight.data = hf_sd[pfx + 'linear_2.weight']
        self.timestep_encoder.linear_2.bias.data   = hf_sd[pfx + 'linear_2.bias']

        # State encoder — read dims from checkpoint
        se_pfx = 'action_head.state_encoder.'
        se_w1 = hf_sd[se_pfx + 'layer1.W']   # [N, in_dim, hidden_dim]
        se_w2 = hf_sd[se_pfx + 'layer2.W']   # [N, hidden_dim, out_dim]
        N, in_dim, h1 = se_w1.shape
        _, h2, out_dim = se_w2.shape
        self.state_encoder = CategorySpecificMLP(N, in_dim, h1, out_dim)
        self.state_encoder.layer1.W.data = se_w1
        self.state_encoder.layer1.b.data = hf_sd[se_pfx + 'layer1.b']
        self.state_encoder.layer2.W.data = se_w2
        self.state_encoder.layer2.b.data = hf_sd[se_pfx + 'layer2.b']

        # Action encoder W1 (action_dim → 1536)
        ae_pfx = 'action_head.action_encoder.'
        ae_w1 = hf_sd[ae_pfx + 'W1.W']  # [N, action_dim, embed_dim]
        _, a_in, a_out = ae_w1.shape
        self.action_encoder_W1 = CategorySpecificLinear(N, a_in, a_out)
        self.action_encoder_W1.W.data = ae_w1
        self.action_encoder_W1.b.data = hf_sd[ae_pfx + 'W1.b']

        # Action decoder — read dims from checkpoint
        ad_pfx = 'action_head.action_decoder.'
        ad_w1 = hf_sd[ad_pfx + 'layer1.W']   # [N, in_dim, hidden_dim]
        ad_w2 = hf_sd[ad_pfx + 'layer2.W']   # [N, hidden_dim, out_dim]
        _, ad_in, ad_h = ad_w1.shape
        _, _, ad_out = ad_w2.shape
        self.action_decoder = CategorySpecificMLP(N, ad_in, ad_h, ad_out)
        self.action_decoder.layer1.W.data = ad_w1
        self.action_decoder.layer1.b.data = hf_sd[ad_pfx + 'layer1.b']
        self.action_decoder.layer2.W.data = ad_w2
        self.action_decoder.layer2.b.data = hf_sd[ad_pfx + 'layer2.b']

        # Position embedding
        self.position_embedding.weight.data = hf_sd['action_head.position_embedding.weight']

        # VLLN
        if 'action_head.vlln.weight' in hf_sd:
            self.vlln.weight.data = hf_sd['action_head.vlln.weight']
            self.vlln.bias.data   = hf_sd['action_head.vlln.bias']

        # Cast all CPU modules to bfloat16
        self.timestep_encoder = self.timestep_encoder.bfloat16().eval()
        self.state_encoder = self.state_encoder.bfloat16().eval()
        self.action_encoder_W1 = self.action_encoder_W1.bfloat16().eval()
        self.action_decoder = self.action_decoder.bfloat16().eval()
        self.position_embedding = self.position_embedding.bfloat16().eval()
        self.vlln = self.vlln.bfloat16().eval()

    @staticmethod
    def _is_on_trainium() -> bool:
        """Check if Trainium hardware is available and NEFF weights are sharded."""
        try:
            # Check if neuron device is actually available
            import subprocess
            result = subprocess.run(
                ['neuron-ls'],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def load_compiled_backbone(self, hf_sd: dict) -> None:
        """Load or build backbone subgraph."""
        neff_path = os.path.join(BACKBONE_DIR, MODEL_FILE)
        if os.path.isfile(neff_path) and self._is_on_trainium():
            # Load from compiled NEFF (Trainium only)
            print(f"Loading backbone NEFF from {BACKBONE_DIR}")
            cfg = _make_backbone_compile_config()
            head = NeuronGR00TBackboneHead(model_path=MODEL_PATH, config=cfg)
            # Build wrapper so head.load() can attach traced_model to it
            wrapper = NeuronGR00TBackboneWrapper()
            head.denoising_wrapper = wrapper
            head.load(BACKBONE_DIR)
            # After load(), wrapper.model = traced_model
            self.backbone_wrapper = wrapper
        else:
            # CPU fallback (testing without Trainium)
            if os.path.isfile(neff_path):
                print("Backbone NEFF found but not on Trainium, using CPU forward")
            else:
                print("Backbone NEFF not found, using CPU forward (for testing only)")
            backbone_sd = load_backbone_state_dict(hf_sd)
            wrapper = NeuronGR00TBackboneWrapper()
            wrapper._preload_sd = {k.removeprefix('model.'): v for k, v in backbone_sd.items()}
            wrapper.load_module()
            self.backbone_wrapper = wrapper

    def load_compiled_vl_self_attn(self, hf_sd: dict) -> None:
        """Load or build VL self-attention subgraph."""
        neff_path = os.path.join(VL_SA_DIR, MODEL_FILE)
        if os.path.isfile(neff_path) and self._is_on_trainium():
            from compile_all import _VLSelfAttnHead, _make_vl_self_attn_compile_config
            print(f"Loading VL self-attn NEFF from {VL_SA_DIR}")
            cfg = _make_vl_self_attn_compile_config()
            head = _VLSelfAttnHead(model_path=MODEL_PATH, config=cfg)
            # Build wrapper so head.load() can attach traced_model to it
            wrapper = NeuronGR00TVLSelfAttnWrapper()
            head.denoising_wrapper = wrapper
            head.load(VL_SA_DIR)
            self.vl_self_attn_wrapper = wrapper
        else:
            if os.path.isfile(neff_path):
                print("VL self-attn NEFF found but not on Trainium, using CPU forward")
            else:
                print("VL self-attn NEFF not found, using CPU forward (for testing only)")
            vl_sd = load_vl_self_attn_state_dict(hf_sd)
            wrapper = NeuronGR00TVLSelfAttnWrapper()
            wrapper._preload_sd = {k.removeprefix('model.'): v for k, v in vl_sd.items()}
            wrapper.load_module()
            self.vl_self_attn_wrapper = wrapper

    def load_compiled_dit(self, hf_sd: dict) -> None:
        """Load or build DiT subgraph."""
        neff_path = os.path.join(DIT_DIR, MODEL_FILE)
        if os.path.isfile(neff_path) and self._is_on_trainium():
            print(f"Loading DiT NEFF from {DIT_DIR}")
            cfg = NeuronDenoisingConfig(
                batch_size=BATCH_SIZE, tp_degree=TP_DEGREE_DIT,
                action_chunk_size=DIT_INPUT_SEQ_LEN, action_dim=DIT_OUTPUT_DIM,
                num_conditioning_tokens=NUM_CONDITIONING_TOKENS,
                conditioning_hidden_size=CONDITIONING_HIDDEN_SIZE,
                timestep_embed_dim=TIMESTEP_EMBED_DIM,
            )
            head = NeuronGR00TActionHead(model_path=MODEL_PATH, config=cfg)
            # Build wrapper so head.load() can attach traced_model to it
            wrapper = NeuronGR00TDenoisingWrapper()
            head.denoising_wrapper = wrapper
            head.load(DIT_DIR)
            self.dit_wrapper = wrapper
        else:
            if os.path.isfile(neff_path):
                print("DiT NEFF found but not on Trainium, using CPU forward")
            else:
                print("DiT NEFF not found, using CPU forward (for testing only)")
            dit_sd = load_dit_state_dict(hf_sd)
            wrapper = NeuronGR00TDenoisingWrapper()
            wrapper._preload_sd = {k.removeprefix('model.'): v for k, v in dit_sd.items()}
            wrapper.load_module()
            self.dit_wrapper = wrapper

    @torch.no_grad()
    def generate_actions(
        self,
        inputs_embeds: torch.Tensor,  # [B, SEQ, 2048] pre-computed by ViT+embed
        pre_cos: torch.Tensor,        # [B, SEQ, 128] HF mRoPE cos, from run_vit_cpu
        pre_sin: torch.Tensor,        # [B, SEQ, 128] HF mRoPE sin, from run_vit_cpu
        state: torch.Tensor,          # [B, 1, MAX_STATE_DIM]
        embodiment_id: torch.Tensor,  # [B] long
        num_steps: int = NUM_INFERENCE_TIMESTEPS,
        profile: bool = False,
    ) -> torch.Tensor:
        """
        Run inference. ViT must be run on CPU before calling this (via run_vit_cpu).
        Returns [B, ACTION_HORIZON, action_dim] BF16 action chunk.

        Args:
            profile: if True, print per-subgraph timing to stdout.
        """
        import time as _time

        def _t():
            return _time.perf_counter() * 1000.0  # ms

        B = inputs_embeds.shape[0]

        # Step 1: Backbone LLM (pre_cos/pre_sin injected via cos_cache/sin_cache kwargs)
        t0 = _t()
        backbone_out = self.backbone_wrapper(inputs_embeds, pre_cos, pre_sin)
        t_backbone = _t() - t0
        # [B, SEQ, 2048]

        # Step 2: VLLN + VL self-attention
        t0 = _t()
        backbone_out = self.vlln(backbone_out)
        conditioning_tokens = self.vl_self_attn_wrapper(backbone_out)
        t_vl_sa = _t() - t0
        # [B, SEQ, 2048]

        # Step 3: State encoding
        state_flat = state.view(B, 1, -1).to(torch.bfloat16)  # [B, 1, MAX_STATE_DIM]
        state_features = self.state_encoder(state_flat, embodiment_id)  # [B, 1, embed]

        # Step 4: Flow-matching denoising loop
        noisy_actions = torch.randn(
            B, ACTION_HORIZON, MAX_ACTION_DIM, dtype=torch.bfloat16
        )
        cross_mask = torch.ones(
            B, 1, DIT_INPUT_SEQ_LEN, NUM_CONDITIONING_TOKENS, dtype=torch.int32
        )

        dit_times = []
        for step in range(num_steps):
            t = 1.0 - step / num_steps
            t_bucket = torch.tensor(
                [int(t * (NUM_TIMESTEP_BUCKETS - 1))], dtype=torch.long
            ).expand(B)

            # Encode noisy actions
            action_features = self.action_encoder_W1(noisy_actions, embodiment_id)
            action_features = F.silu(action_features)

            # Add position embedding
            pos_ids = torch.arange(action_features.shape[1])
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            # sa_embs = state + action [B, 41, 1536]
            sa_embs = torch.cat([state_features, action_features], dim=1)

            # Timestep embedding
            temb = self.timestep_encoder(t_bucket)  # [B, 1536]

            # DiT forward (compiled NEFF)
            t0 = _t()
            velocity_pred = self.dit_wrapper(sa_embs, conditioning_tokens, temb, cross_mask)
            dit_times.append(_t() - t0)
            # [B, 41, 1024] → action portion [B, 40, 1024]
            action_velocity = velocity_pred[:, -ACTION_HORIZON:, :]

            # Euler step: decode velocity to action space
            dt = 1.0 / num_steps
            v_decoded = self.action_decoder(action_velocity, embodiment_id)
            noisy_actions = noisy_actions - dt * v_decoded

        if profile:
            print(f"  [profile] backbone NEFF:    {t_backbone:.2f}ms")
            print(f"  [profile] VL self-attn NEFF:{t_vl_sa:.2f}ms")
            for i, t in enumerate(dit_times):
                print(f"  [profile] DiT step {i+1}/{num_steps}:   {t:.2f}ms")
            print(f"  [profile] DiT total:        {sum(dit_times):.2f}ms")

        return noisy_actions  # [B, ACTION_HORIZON, MAX_ACTION_DIM]


def load_model() -> GR00TModel:
    """Load all compiled NEFFs + CPU weights. Returns ready-to-use GR00TModel."""
    model = GR00TModel()
    hf_sd = model._load_hf_sd()

    print("Loading CPU weights...")
    model.load_cpu_weights(hf_sd)

    print("Loading backbone subgraph...")
    model.load_compiled_backbone(hf_sd)

    print("Loading VL self-attention subgraph...")
    model.load_compiled_vl_self_attn(hf_sd)

    print("Loading DiT subgraph...")
    model.load_compiled_dit(hf_sd)

    print("Model loaded.")
    return model


# ---------------------------------------------------------------------------
# Main — smoke test with dummy inputs
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("GR00T N1.7 Trainium Inference Smoke Test")
    print("=" * 50)

    model = load_model()
    hf_sd = model._hf_sd

    # Build dummy visual inputs and run ViT on CPU
    print("\nRunning ViT preprocessing (CPU)...")
    backbone_in = make_backbone_inputs(B=BATCH_SIZE, n_text=16)
    t0 = time.time()
    inputs_embeds, pre_cos, pre_sin = run_vit_cpu(
        backbone_in['input_ids'], backbone_in['attention_mask'],
        backbone_in['pixel_values'], backbone_in['image_grid_thw'],
        hf_sd=hf_sd,
    )
    print(f"  ViT done in {time.time()-t0:.2f}s")
    print(f"  inputs_embeds: {inputs_embeds.shape}, pre_cos: {pre_cos.shape}")

    # Dummy state and embodiment
    state = torch.zeros(BATCH_SIZE, 1, MAX_STATE_DIM, dtype=torch.bfloat16)
    embodiment_id = torch.zeros(BATCH_SIZE, dtype=torch.long)

    print("\nRunning full inference...")
    t0 = time.time()
    with torch.no_grad():
        actions = model.generate_actions(
            inputs_embeds=inputs_embeds,
            pre_cos=pre_cos,
            pre_sin=pre_sin,
            state=state,
            embodiment_id=embodiment_id,
            num_steps=NUM_INFERENCE_TIMESTEPS,
        )
    elapsed = time.time() - t0

    print(f"  Inference done in {elapsed:.3f}s")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Actions stats: mean={actions.float().mean():.4f}, std={actions.float().std():.4f}")
    assert actions.shape == (BATCH_SIZE, ACTION_HORIZON, MAX_ACTION_DIM), \
        f"Unexpected action shape: {actions.shape}"
    assert not torch.isnan(actions).any(), "NaN in output actions!"
    print("\nSmoke test PASSED.")
