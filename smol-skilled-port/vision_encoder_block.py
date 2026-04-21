"""
Vision Encoder Block for SmolVLA port to AWS Trainium (NxDI).

Implements SigLIP ViT (12-layer) + pixel-shuffle connector.

Input:  pixel_values [B, 3, 512, 512] BF16
Output: vision_embeddings [B, 64, 960] BF16
"""
import sys
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

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

from config_constants import (
    VISION_HIDDEN_SIZE,
    VISION_NUM_HEADS,
    VISION_HEAD_DIM,
    VISION_INTERMEDIATE_SIZE,
    VISION_NUM_LAYERS,
    VISION_PATCH_SIZE,
    VISION_IMAGE_SIZE,
    VISION_NUM_PATCHES,
    VISION_PIXEL_SHUFFLE_SCALE,
    VISION_TOKENS_PER_CAMERA,
    VISION_CONNECTOR_INPUT_DIM,
    VISION_CONNECTOR_OUTPUT_DIM,
)


# ---------------------------------------------------------------------------
# SigLIP attention — full MHA WITH biases (not GQA, no RoPE)
# ---------------------------------------------------------------------------
class SigLIPAttention(nn.Module):
    def __init__(
        self,
        hidden_size=VISION_HIDDEN_SIZE,
        num_heads=VISION_NUM_HEADS,
        head_dim=VISION_HEAD_DIM,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        if _is_parallel():
            g = _tp_group()
            self.q_proj   = ColumnParallelLinear(hidden_size, num_heads * head_dim, bias=True, gather_output=True,  dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.k_proj   = ColumnParallelLinear(hidden_size, num_heads * head_dim, bias=True, gather_output=True,  dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.v_proj   = ColumnParallelLinear(hidden_size, num_heads * head_dim, bias=True, gather_output=True,  dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.out_proj = RowParallelLinear(num_heads * head_dim, hidden_size,    bias=True, input_is_parallel=False, dtype=torch.bfloat16, tensor_model_parallel_group=g)
        else:
            self.q_proj   = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
            self.k_proj   = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
            self.v_proj   = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
            self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=True)

    def forward(self, hidden_states):
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)

        # [B, H, L, D]
        q = q.transpose(1, 2).to(torch.float32)
        k = k.transpose(1, 2).to(torch.float32)
        v = v.transpose(1, 2).to(torch.float32)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(2, 3)) * scale
        attn = F.softmax(attn, dim=-1).to(hidden_states.dtype)

        out = torch.matmul(attn, v.to(hidden_states.dtype))
        out = out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# SigLIP encoder layer — LayerNorm (not RMSNorm), GELU MLP
# ---------------------------------------------------------------------------
class SigLIPMLP(nn.Module):
    def __init__(self, hidden_size=VISION_HIDDEN_SIZE, intermediate_size=VISION_INTERMEDIATE_SIZE):
        super().__init__()
        if _is_parallel():
            g = _tp_group()
            self.fc1 = ColumnParallelLinear(hidden_size, intermediate_size, bias=True, gather_output=False, dtype=torch.bfloat16, tensor_model_parallel_group=g)
            self.fc2 = RowParallelLinear(intermediate_size, hidden_size,    bias=True, input_is_parallel=True, dtype=torch.bfloat16, tensor_model_parallel_group=g)
        else:
            self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
            self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SigLIPEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=VISION_HIDDEN_SIZE,
        num_heads=VISION_NUM_HEADS,
        head_dim=VISION_HEAD_DIM,
        intermediate_size=VISION_INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn   = SigLIPAttention(hidden_size, num_heads, head_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp         = SigLIPMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Patch embedding + position embedding
# ---------------------------------------------------------------------------
class SigLIPEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size=VISION_HIDDEN_SIZE,
        patch_size=VISION_PATCH_SIZE,
        image_size=VISION_IMAGE_SIZE,
        num_patches=VISION_NUM_PATCHES,
    ):
        super().__init__()
        self.patch_embedding    = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.position_embedding = nn.Embedding(num_patches, hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(num_patches).expand(1, -1),
            persistent=False,
        )

    def forward(self, pixel_values):
        # pixel_values: [B, 3, H, W] → [B, 768, 32, 32] → [B, 1024, 768]
        pixel_values = pixel_values.to(self.patch_embedding.weight.dtype)
        patches = self.patch_embedding(pixel_values)
        B, C, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)   # [B, 1024, 768]
        pos_emb = self.position_embedding(self.position_ids)  # [1, 1024, 768]
        return patches + pos_emb


# ---------------------------------------------------------------------------
# Pixel-shuffle connector (verbatim from SmolVLMConnector)
# ---------------------------------------------------------------------------
def pixel_shuffle(x, scale_factor):
    bsz, seq, embed_dim = x.size()
    height = width = int(seq ** 0.5)
    x = x.view(bsz, height, width, embed_dim)
    x = x.view(bsz, height, width // scale_factor, embed_dim * scale_factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bsz, width // scale_factor, height // scale_factor, embed_dim * (scale_factor ** 2))
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bsz, seq // (scale_factor ** 2), embed_dim * (scale_factor ** 2))
    return x


class SmolVLMConnector(nn.Module):
    def __init__(
        self,
        input_dim=VISION_CONNECTOR_INPUT_DIM,
        output_dim=VISION_CONNECTOR_OUTPUT_DIM,
        scale_factor=VISION_PIXEL_SHUFFLE_SCALE,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        if _is_parallel():
            g = _tp_group()
            self.proj = ColumnParallelLinear(input_dim, output_dim, bias=False, gather_output=True, dtype=torch.bfloat16, tensor_model_parallel_group=g)
        else:
            self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        x = pixel_shuffle(x, self.scale_factor)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Main NeuronVisionEncoderBlock
# ---------------------------------------------------------------------------
class NeuronVisionEncoderBlock(nn.Module):
    """
    SigLIP ViT + SmolVLM connector.
    Input:  pixel_values [B, 3, 512, 512] BF16
    Output: vision_embeddings [B, 64, 960] BF16
    """

    def __init__(self, config=None):
        super().__init__()
        self.embeddings = SigLIPEmbeddings()
        self.encoder    = nn.ModuleList([SigLIPEncoderLayer() for _ in range(VISION_NUM_LAYERS)])
        self.post_layernorm = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=1e-6)
        self.connector  = SmolVLMConnector()

    def forward(self, pixel_values):
        """pixel_values: [B, 3, 512, 512] → [B, 64, 960]"""
        hidden = self.embeddings(pixel_values)
        for layer in self.encoder:
            hidden = layer(hidden)
        hidden = self.post_layernorm(hidden)           # [B, 1024, 768]
        return self.connector(hidden)                  # [B, 64, 960]


# ---------------------------------------------------------------------------
# Weight mapping: ref_state_dict_key → neuron_block_state_dict_key
# Reference is SmolVisionEncoderReference whose .model = SmolVLMWithExpertModel
# so ref keys start with "model.vlm.model."
# ---------------------------------------------------------------------------
def get_vision_weight_mapping():
    """
    Returns dict mapping reference state_dict key → NeuronVisionEncoderBlock param name.
    Used by test_block_correctness which iterates (ref_key, neuron_key).
    """
    mapping = {}
    ref_base = "model.vlm.model."

    # Embeddings
    mapping[ref_base + "vision_model.embeddings.patch_embedding.weight"]    = "embeddings.patch_embedding.weight"
    mapping[ref_base + "vision_model.embeddings.patch_embedding.bias"]      = "embeddings.patch_embedding.bias"
    mapping[ref_base + "vision_model.embeddings.position_embedding.weight"] = "embeddings.position_embedding.weight"

    # Encoder layers
    for i in range(VISION_NUM_LAYERS):
        r = ref_base + f"vision_model.encoder.layers.{i}"
        p = f"encoder.{i}"
        for name in ["layer_norm1.weight", "layer_norm1.bias",
                     "layer_norm2.weight", "layer_norm2.bias",
                     "self_attn.q_proj.weight", "self_attn.q_proj.bias",
                     "self_attn.k_proj.weight", "self_attn.k_proj.bias",
                     "self_attn.v_proj.weight", "self_attn.v_proj.bias",
                     "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                     "mlp.fc1.weight", "mlp.fc1.bias",
                     "mlp.fc2.weight", "mlp.fc2.bias"]:
            mapping[f"{r}.{name}"] = f"{p}.{name}"

    # post_layernorm
    mapping[ref_base + "vision_model.post_layernorm.weight"] = "post_layernorm.weight"
    mapping[ref_base + "vision_model.post_layernorm.bias"]   = "post_layernorm.bias"

    # Connector
    mapping[ref_base + "connector.modality_projection.proj.weight"] = "connector.proj.weight"

    return mapping


# ---------------------------------------------------------------------------
# PyTorch reference — wraps SmolVLMWithExpertModel.embed_image()
# ---------------------------------------------------------------------------
class SmolVisionEncoderReference(nn.Module):
    """
    Reference using the installed lerobot SmolVLMWithExpertModel.embed_image().
    Accepts a SINGLE pixel_values tensor (single-arg per test harness convention).
    """

    def __init__(self):
        super().__init__()
        from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
        self.model = SmolVLMWithExpertModel(load_vlm_weights=False)
        vlm = self.model.get_vlm_model()
        vlm.vision_model._attn_implementation = "eager"

    def forward(self, pixel_values):
        # embed_image returns [B, 64, 960]; cast to input dtype to match neuron block
        return self.model.embed_image(pixel_values).to(pixel_values.dtype)
