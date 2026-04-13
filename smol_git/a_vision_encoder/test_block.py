"""
Unit test for NeuronSmolVLMVisionEncoder vs Idefics3VisionTransformer reference.
"""
import sys
sys.path.insert(0, '/home/ubuntu/smol_port/tests')
import _transformers_compat  # bypass huggingface-hub version check — must be first

import torch
import torch.nn as nn
from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionTransformer
from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig

from block_testing_utils import test_block_correctness
from nxdi_block import NeuronSmolVLMVisionEncoder


class Idefics3VisionTransformerWrapper(nn.Module):
    """
    Thin wrapper around Idefics3VisionTransformer that returns a plain tensor
    (last_hidden_state) instead of a BaseModelOutput dataclass.
    The testing utility calls reference_block(pixel_values) and expects a tensor.
    """

    def __init__(self, config):
        super().__init__()
        self.model = Idefics3VisionTransformer(config)

    def forward(self, pixel_values):
        out = self.model(pixel_values)
        return out.last_hidden_state


# --- Architecture constants ---
HIDDEN_SIZE = 768
IMAGE_SIZE = 512
PATCH_SIZE = 16
NUM_HEADS = 12
INTERMEDIATE_SIZE = 3072
NUM_LAYERS = 27
LAYER_NORM_EPS = 1e-6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 1024
BATCH_SIZE = 1


def build_weight_mapping(num_layers: int = NUM_LAYERS):
    """
    Map Idefics3VisionTransformerWrapper state_dict keys -> NeuronSmolVLMVisionEncoder keys.

    Wrapper state_dict has prefix "model." (the inner Idefics3VisionTransformer).
    Reference inner model uses: model.encoder.layers.N.*
    NxDI uses:                  encoder.N.*
    """
    mapping = {}

    # Patch embedding (no bias in source Conv2d)
    mapping["model.embeddings.patch_embedding.weight"] = "embeddings.patch_embedding.weight"

    # Position embedding
    mapping["model.embeddings.position_embedding.weight"] = "embeddings.position_embedding.weight"

    # Encoder layers
    for i in range(num_layers):
        ref_prefix = f"model.encoder.layers.{i}"
        nxd_prefix = f"encoder.{i}"

        # Layer norms
        for ln in ("layer_norm1", "layer_norm2"):
            for param in ("weight", "bias"):
                mapping[f"{ref_prefix}.{ln}.{param}"] = f"{nxd_prefix}.{ln}.{param}"

        # Attention projections
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            for param in ("weight", "bias"):
                mapping[f"{ref_prefix}.self_attn.{proj}.{param}"] = (
                    f"{nxd_prefix}.self_attn.{proj}.{param}"
                )

        # MLP
        for fc in ("fc1", "fc2"):
            for param in ("weight", "bias"):
                mapping[f"{ref_prefix}.mlp.{fc}.{param}"] = f"{nxd_prefix}.mlp.{fc}.{param}"

    # Post layer norm
    mapping["model.post_layernorm.weight"] = "post_layernorm.weight"
    mapping["model.post_layernorm.bias"] = "post_layernorm.bias"

    return mapping


def test_vision_encoder():
    vision_cfg = Idefics3VisionConfig(
        hidden_size=HIDDEN_SIZE,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act="gelu",
        layer_norm_eps=LAYER_NORM_EPS,
        num_hidden_layers=NUM_LAYERS,
    )

    pixel_values = torch.zeros(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bfloat16)
    sample = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bfloat16)

    weight_mapping = build_weight_mapping(NUM_LAYERS)

    test_block_correctness(
        neuron_block_class=NeuronSmolVLMVisionEncoder,
        pytorch_block_class=Idefics3VisionTransformerWrapper,
        weight_mapping=weight_mapping,
        neuron_init_kwargs={
            "hidden_size": HIDDEN_SIZE,
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "num_attention_heads": NUM_HEADS,
            "num_hidden_layers": NUM_LAYERS,
            "intermediate_size": INTERMEDIATE_SIZE,
            "layer_norm_eps": LAYER_NORM_EPS,
        },
        pytorch_init_kwargs={
            "config": vision_cfg,
        },
        example_inputs=[(pixel_values,)],
        test_inputs=[(sample,)],
        reference_inputs=[(sample,)],
        checkpoint_name="vision_encoder.pt",
        batch_size=BATCH_SIZE,
        seq_len=NUM_PATCHES,
        hidden_size=HIDDEN_SIZE,
        verbose=True,
        # 27 bfloat16 transformer layers accumulate ~0.03 relative error — relax rtol accordingly
        assert_close_kwargs={"rtol": 0.05, "atol": 0.01},
    )
    print("PASSED: NeuronSmolVLMVisionEncoder matches Idefics3VisionTransformer")


if __name__ == "__main__":
    test_vision_encoder()
