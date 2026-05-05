"""
pi0_neuron.py — Full π0 inference pipeline on AWS Trainium.

Three compiled subgraphs wired together:
  1. vision_encoder.pt  — SigLIP + projector (per-camera, TP=1)
  2. prefix_encoder.pt  — Gemma 2B backbone, outputs KV cache (TP=1)
  3. suffix_denoiser.pt — Gemma 300M action expert, 10× flow-matching (TP=1)

Usage:
    # Load compiled NEFFs (after compile step)
    model = Pi0NeuronModel.load(
        checkpoint_path="/home/ubuntu/pi0-port/weights",
        compiled_dir="/home/ubuntu/pi0-port/compiled",
    )
    actions = model.generate_actions(images, lang_tokens, lang_masks, state)

    # Or compile from scratch:
    Pi0NeuronModel.compile(
        checkpoint_path="/home/ubuntu/pi0-port/weights",
        compiled_dir="/home/ubuntu/pi0-port/compiled",
    )
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/ubuntu/pi0-port')
sys.path.insert(0, '/home/ubuntu/pi0-port/skills/scripts')

from config_constants import (
    SIGLIP_NUM_IMAGE_TOKENS, VLM_HIDDEN_SIZE,
    SIGLIP_IMAGE_SIZE, NUM_CAMERAS, MAX_LANG_TOKENS,
    PREFIX_LEN, CHUNK_SIZE, MAX_STATE_DIM, MAX_ACTION_DIM,
    EXPERT_HIDDEN_SIZE, NUM_INFERENCE_STEPS,
    ACTUAL_STATE_DIM, ACTUAL_ACTION_DIM, TARGET_IMAGE_H, TARGET_IMAGE_W,
)


# ── Language token embedding ──────────────────────────────────────────────────

class LanguageEmbedder(nn.Module):
    """Wraps the PaliGemma embed_tokens layer for language token embedding."""

    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.hidden_size = VLM_HIDDEN_SIZE

    def forward(self, lang_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lang_tokens: [B, T_lang] int64
        Returns:
            lang_embs: [B, T_lang, 2048] bfloat16
        """
        emb = self.embed_tokens(lang_tokens)            # [B, T, 2048]
        emb = emb * math.sqrt(self.hidden_size)         # scale as in embed_prefix
        return emb.to(torch.bfloat16)


# ── Pi0NeuronModel ────────────────────────────────────────────────────────────

class Pi0NeuronModel:
    """
    Full π0 inference pipeline on Trainium.

    Not a torch.nn.Module — manages three compiled NEFFs + CPU preprocessing.
    """

    def __init__(
        self,
        vision_encoder,        # NeuronVisionEncoder (traced or CPU)
        language_embedder,     # LanguageEmbedder
        prefix_encoder,        # NeuronPrefixEncoder (traced or CPU)
        action_head,           # NeuronPi0ActionHead (with compiled denoiser)
        num_cameras: int = NUM_CAMERAS,
        actual_action_dim: int = ACTUAL_ACTION_DIM,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
    ):
        self.vision_encoder = vision_encoder
        self.language_embedder = language_embedder
        self.prefix_encoder = prefix_encoder
        self.action_head = action_head
        self.num_cameras = num_cameras
        self.actual_action_dim = actual_action_dim
        self.num_inference_steps = num_inference_steps

    def _embed_prefix(
        self,
        images: list,           # list of NUM_CAMERAS tensors, each [B, 3, H, W] float32
        lang_tokens: torch.Tensor,   # [B, T_lang] int64
    ) -> torch.Tensor:
        """
        Build prefix_embs [B, PREFIX_LEN, 2048] bfloat16.
        Embeds images through SigLIP and language tokens through embed_tokens.
        """
        assert len(images) == self.num_cameras

        # Embed each camera image
        img_embs = []
        for img in images:
            # img: [B, 3, H, W] float32, normalized to [-1, 1]
            emb = self.vision_encoder(img)              # [B, 256, 2048] float32
            emb = emb * math.sqrt(VLM_HIDDEN_SIZE)      # scale
            img_embs.append(emb.to(torch.bfloat16))    # [B, 256, 2048] bfloat16

        # Language token embeddings
        lang_embs = self.language_embedder(lang_tokens)  # [B, T_lang, 2048] bfloat16

        # Concatenate: [B, 3*256, 2048] + [B, T_lang, 2048] → [B, 816, 2048]
        prefix_embs = torch.cat(img_embs + [lang_embs], dim=1)
        return prefix_embs

    @torch.no_grad()
    def generate_actions(
        self,
        images: list,              # list of NUM_CAMERAS tensors [B, 3, H, W] float32 in [-1,1]
        lang_tokens: torch.Tensor, # [B, T_lang] int64, padded to MAX_LANG_TOKENS
        lang_masks: torch.Tensor,  # [B, T_lang] bool — currently unused (all positions valid)
        state: torch.Tensor,       # [B, state_dim] float32
    ) -> torch.Tensor:             # [B, CHUNK_SIZE, actual_action_dim] float32
        """
        Run the full π0 inference pipeline:
          1. Embed images + language tokens → prefix_embs
          2. Run prefix encoder → prefix KV cache
          3. Run 10-step flow-matching denoising loop → actions
        """
        # 1. Embed prefix
        prefix_embs = self._embed_prefix(images, lang_tokens)  # [B, 816, 2048]

        # 2. Prefix encoder → KV cache
        prefix_kv = self.prefix_encoder(prefix_embs).to(torch.bfloat16)  # [18, 2, B, 816, 1, 256]

        # 3. Pad state to MAX_STATE_DIM
        B = state.shape[0]
        state_padded = state.to(torch.bfloat16)
        if state_padded.shape[-1] < MAX_STATE_DIM:
            state_padded = F.pad(state_padded, (0, MAX_STATE_DIM - state_padded.shape[-1]))

        # 4. Flow-matching denoising loop
        actions = self.action_head.generate_actions_with_kv(
            prefix_kv=prefix_kv,
            state=state_padded,
            num_steps=self.num_inference_steps,
            actual_action_dim=self.actual_action_dim,
        )

        return actions.to(torch.float32)  # [B, 50, 6]

    @classmethod
    def compile(
        cls,
        checkpoint_path: str,
        compiled_dir: str,
        num_cameras: int = NUM_CAMERAS,
    ):
        """Compile all three subgraphs to NEFFs."""
        from vision_encoder import load_vision_encoder, compile_vision_encoder
        from prefix_encoder import load_prefix_encoder, compile_prefix_encoder
        from suffix_denoiser import NeuronPi0ActionHead

        vis_path = os.path.join(compiled_dir, "vision_encoder/")
        pre_path = os.path.join(compiled_dir, "prefix_encoder/")
        suf_path = os.path.join(compiled_dir, "suffix_denoiser/")
        ckpt_file = os.path.join(checkpoint_path, "model.safetensors")

        print("=== Compiling Vision Encoder ===")
        ve = load_vision_encoder(ckpt_file)
        compile_vision_encoder(ve, vis_path)

        print("=== Compiling Prefix Encoder ===")
        pe = load_prefix_encoder(ckpt_file)
        compile_prefix_encoder(pe, pre_path)

        print("=== Compiling Suffix Denoiser ===")
        head = NeuronPi0ActionHead(model_path=ckpt_file)
        head.compile_denoiser(save_path=suf_path)

        print("All subgraphs compiled.")

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        compiled_dir: str,
        actual_action_dim: int = ACTUAL_ACTION_DIM,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
    ) -> "Pi0NeuronModel":
        """Load compiled NEFFs and weights for inference."""
        import torch_neuronx
        from vision_encoder import NeuronVisionEncoder, _build_paligemma_model
        from prefix_encoder import NeuronPrefixEncoder, build_gemma_model
        from suffix_denoiser import NeuronPi0ActionHead
        from safetensors.torch import load_file

        ckpt_file = os.path.join(checkpoint_path, "model.safetensors")
        vis_path = os.path.join(compiled_dir, "vision_encoder/")
        pre_path = os.path.join(compiled_dir, "prefix_encoder/")
        suf_path = os.path.join(compiled_dir, "suffix_denoiser/")

        # ── Vision encoder ─────────────────────────────────────────────────
        print("Loading vision encoder...")
        full_model = _build_paligemma_model()
        sd = load_file(ckpt_file)
        vis_prefix = "model.paligemma_with_expert.paligemma.model.vision_tower."
        prj_prefix = "model.paligemma_with_expert.paligemma.model.multi_modal_projector."
        vis_sd = {k[len(vis_prefix):]: v for k, v in sd.items() if k.startswith(vis_prefix)}
        prj_sd = {k[len(prj_prefix):]: v for k, v in sd.items() if k.startswith(prj_prefix)}
        full_model.model.vision_tower.load_state_dict(vis_sd, strict=True)
        full_model.model.multi_modal_projector.load_state_dict(prj_sd, strict=True)
        ve = NeuronVisionEncoder(full_model.model.vision_tower, full_model.model.multi_modal_projector)
        ve_neff = torch.jit.load(os.path.join(vis_path, "model.pt"))
        # The NEFF captures the full forward (vision_tower + projector) as one graph.
        # Replace the entire forward, not just vision_tower.
        ve._neff = ve_neff
        import types
        ve.forward = types.MethodType(lambda self, x: self._neff(x), ve)

        # ── Language embedder (CPU, not compiled) ─────────────────────────
        # embed_tokens.weight is tied to lm_head.weight in this checkpoint (no separate key).
        lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
        full_model.model.language_model.embed_tokens.load_state_dict(
            {"weight": sd[lm_head_key]}, strict=True
        )
        lang_emb = LanguageEmbedder(full_model.model.language_model.embed_tokens)

        # ── Prefix encoder ─────────────────────────────────────────────────
        print("Loading prefix encoder...")
        pe = NeuronPrefixEncoder(build_gemma_model())
        pe_neff = torch.jit.load(os.path.join(pre_path, "model.pt"))
        pe._neff = pe_neff
        import types
        pe.forward = types.MethodType(lambda self, x: self._neff(x), pe)

        # ── Suffix denoiser ────────────────────────────────────────────────
        # NeuronPi0ActionHead.load() handles NEFF + sharded weight initialization.
        # model_path must be the DIRECTORY containing model.safetensors.
        print("Loading suffix denoiser...")
        head = NeuronPi0ActionHead(model_path=checkpoint_path)
        head.load(compiled_model_path=suf_path)

        return cls(
            vision_encoder=ve,
            language_embedder=lang_emb,
            prefix_encoder=pe,
            action_head=head,
            actual_action_dim=actual_action_dim,
            num_inference_steps=num_inference_steps,
        )


# ── Image preprocessing ────────────────────────────────────────────────────────

def preprocess_image(img_tensor: torch.Tensor, target_h: int = TARGET_IMAGE_H,
                     target_w: int = TARGET_IMAGE_W) -> torch.Tensor:
    """
    Resize and pad image to target size, normalize to [-1, 1].

    Args:
        img_tensor: [B, C, H, W] float32 in [0, 1]
    Returns:
        [B, C, target_h, target_w] float32 in [-1, 1]
    """
    B, C, H, W = img_tensor.shape
    ratio = max(W / target_w, H / target_h)
    new_h, new_w = int(H / ratio), int(W / ratio)
    resized = F.interpolate(img_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h0 = (target_h - new_h) // 2
    pad_h1 = target_h - new_h - pad_h0
    pad_w0 = (target_w - new_w) // 2
    pad_w1 = target_w - new_w - pad_w0
    padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), value=0.0)
    return padded * 2.0 - 1.0   # [0,1] → [-1,1]
