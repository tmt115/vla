"""
vision_encoder.py — π0 SigLIP vision encoder + multi-modal projector for Trainium.

Input:  pixel_values [B, 3, 224, 224] float32, normalized to [-1, 1]
Output: image_features [B, 256, 2048] float32
        (caller multiplies by sqrt(2048) before concatenating with language embeddings)

Note: PaliGemmaModel.get_image_features() uses last_hidden_state (all 256 patch tokens)
and then applies multi_modal_projector followed by division by sqrt(hidden_size).
This module returns the projected features WITHOUT the sqrt scaling so that the
caller can apply it at the concatenation site, matching the original pi0 flow.
"""
import math
import os
import torch
import torch.nn as nn
from safetensors.torch import load_file


def _build_paligemma_model():
    """Build a PaliGemmaForConditionalGenerationWithPiGemma shell with correct config."""
    from transformers.models.auto import CONFIG_MAPPING
    from lerobot.policies.pi_gemma import PaliGemmaForConditionalGenerationWithPiGemma

    vlm_config = CONFIG_MAPPING["paligemma"]()
    vlm_config._vocab_size = 257152
    vlm_config.image_token_index = 257152
    vlm_config.text_config.hidden_size = 2048
    vlm_config.text_config.intermediate_size = 16384
    vlm_config.text_config.num_attention_heads = 8
    vlm_config.text_config.head_dim = 256
    vlm_config.text_config.num_hidden_layers = 18
    vlm_config.text_config.num_key_value_heads = 1
    vlm_config.text_config.hidden_activation = "gelu_pytorch_tanh"
    vlm_config.text_config.dtype = "float32"
    vlm_config.text_config.vocab_size = 257152
    vlm_config.text_config.use_adarms = False
    vlm_config.text_config.adarms_cond_dim = None
    vlm_config.vision_config.image_size = 224
    vlm_config.vision_config.intermediate_size = 4304
    vlm_config.vision_config.projection_dim = 2048
    vlm_config.vision_config.projector_hidden_act = "gelu_fast"
    vlm_config.vision_config.dtype = "float32"

    return PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config)


class NeuronVisionEncoder(nn.Module):
    """SigLIP ViT + multi-modal projector, ready for torch_neuronx.trace().

    forward() replicates the first two steps of PaliGemmaModel.get_image_features():
      1. vision_tower(pixel_values) → last_hidden_state  [B, 256, 1152]
      2. multi_modal_projector(last_hidden_state)         [B, 256, 2048]

    The third step (divide by sqrt(hidden_size)) is intentionally left to the caller
    so this module can be traced independently and the scaling applied at the
    embedding-concatenation site.
    """

    def __init__(self, vision_tower, multi_modal_projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector
        # Required for Neuron tracing — SDPA fails on Trainium
        self.vision_tower.config._attn_implementation = "eager"
        # Disable gradient checkpointing
        if hasattr(self.vision_tower, 'vision_model') and hasattr(self.vision_tower.vision_model, 'encoder'):
            self.vision_tower.vision_model.encoder.gradient_checkpointing = False
        # Keep everything in float32 (pi0 dtype policy keeps vision in float32)
        self.vision_tower = self.vision_tower.float()
        self.multi_modal_projector = self.multi_modal_projector.float()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, 3, 224, 224] float32
        vision_out = self.vision_tower(pixel_values, output_hidden_states=False)
        # Use last_hidden_state (all 256 patch tokens), NOT pooler_output.
        # Confirmed from PaliGemmaModel.get_image_features() line:
        #   selected_image_feature = image_outputs.last_hidden_state
        patch_features = vision_out.last_hidden_state  # [B, 256, 1152]
        projected = self.multi_modal_projector(patch_features)  # [B, 256, 2048]
        # NOTE: get_image_features() also divides by sqrt(hidden_size=2048) here,
        # but that step is left to the caller (matches pi0 embedding concat site).
        return projected


def load_vision_encoder(checkpoint_path: str) -> NeuronVisionEncoder:
    """Load SigLIP + projector weights from the lerobot/pi0 checkpoint."""
    full_model = _build_paligemma_model()
    sd = load_file(checkpoint_path)

    # Extract and remap weights
    vision_sd = {}
    proj_sd = {}
    vis_prefix = "model.paligemma_with_expert.paligemma.model.vision_tower."
    proj_prefix = "model.paligemma_with_expert.paligemma.model.multi_modal_projector."
    for k, v in sd.items():
        if k.startswith(vis_prefix):
            vision_sd[k[len(vis_prefix):]] = v
        elif k.startswith(proj_prefix):
            proj_sd[k[len(proj_prefix):]] = v

    full_model.model.vision_tower.load_state_dict(vision_sd, strict=True)
    full_model.model.multi_modal_projector.load_state_dict(proj_sd, strict=True)

    return NeuronVisionEncoder(full_model.model.vision_tower, full_model.model.multi_modal_projector)


def compile_vision_encoder(encoder: NeuronVisionEncoder, save_path: str) -> None:
    import torch_neuronx
    os.makedirs(save_path, exist_ok=True)
    neff_path = os.path.join(save_path, "model.pt")
    if os.path.exists(neff_path):
        print(f"Already compiled, skipping: {neff_path}")
        return
    example = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    encoder.eval()
    traced = torch_neuronx.trace(
        encoder,
        example,
        compiler_workdir=os.path.join(save_path, "workdir/"),
        compiler_args=[
            "--auto-cast", "matmult",
            "--optlevel", "3",
            "--model-type", "unet-inference",
        ],
    )
    torch.jit.save(traced, neff_path)
    sz = os.path.getsize(neff_path) / 1e6
    print(f"Saved: {neff_path} ({sz:.1f} MB)")
