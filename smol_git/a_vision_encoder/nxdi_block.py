"""
NxDI Vision Encoder Block for SmolVLM2-500M (Idefics3 SigLIP ViT-L/16).

Translates Idefics3VisionTransformer into a Neuron/NxDI-compatible module.
When tensor parallelism is not initialized (e.g. CPU tests), falls back to
plain nn.Linear.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    HAS_NXD = True
except Exception:
    HAS_NXD = False


class NeuronSmolVLMVisionAttention(nn.Module):
    """
    Multi-head self-attention (non-causal, no RoPE).
    hidden_size=768, num_heads=12, head_dim=64.
    Uses ColumnParallelLinear for q/k/v and RowParallelLinear for out_proj
    when TP is initialized; plain nn.Linear otherwise.
    """

    def __init__(self, hidden_size: int = 768, num_heads: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        if HAS_NXD and parallel_state.model_parallel_is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()
            self.q_proj = ColumnParallelLinear(
                hidden_size, hidden_size,
                bias=True,
                gather_output=False,
                dtype=torch.bfloat16,

                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=tp_group,
            )
            self.k_proj = ColumnParallelLinear(
                hidden_size, hidden_size,
                bias=True,
                gather_output=False,
                dtype=torch.bfloat16,

                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=tp_group,
            )
            self.v_proj = ColumnParallelLinear(
                hidden_size, hidden_size,
                bias=True,
                gather_output=False,
                dtype=torch.bfloat16,

                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=tp_group,
            )
            self.out_proj = RowParallelLinear(
                hidden_size, hidden_size,
                bias=True,
                input_is_parallel=True,
                dtype=torch.bfloat16,

                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [B, num_heads, S, head_dim]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (non-causal, no mask)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads: [B, num_heads, S, head_dim] -> [B, S, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        return attn_output


class NeuronSmolVLMVisionMLP(nn.Module):
    """
    Two-layer MLP: hidden_size=768 -> intermediate_size=3072 -> hidden_size=768.
    GELU activation. ColumnParallelLinear for fc1, RowParallelLinear for fc2.
    """

    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072):
        super().__init__()

        if HAS_NXD and parallel_state.model_parallel_is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()
            self.fc1 = ColumnParallelLinear(
                hidden_size, intermediate_size,
                bias=True,
                gather_output=False,
                dtype=torch.bfloat16,

                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=tp_group,
            )
            self.fc2 = RowParallelLinear(
                intermediate_size, hidden_size,
                bias=True,
                input_is_parallel=True,
                dtype=torch.bfloat16,

                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.fc1 = nn.Linear(hidden_size, intermediate_size)
            self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class NeuronSmolVLMVisionEncoderLayer(nn.Module):
    """Single Transformer encoder layer (pre-norm, non-causal)."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 12,
                 intermediate_size: int = 3072, layer_norm_eps: float = 1e-6):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = NeuronSmolVLMVisionAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = NeuronSmolVLMVisionMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states).to(dtype=residual.dtype)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with pre-norm and residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states).to(dtype=residual.dtype)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class NeuronSmolVLMVisionEmbeddings(nn.Module):
    """
    Patch + position embeddings for the vision encoder.
    image_size=512, patch_size=16 -> num_patches=1024.
    Position IDs are computed dynamically (same as Idefics3VisionEmbeddings).
    """

    def __init__(self, hidden_size: int = 768, image_size: int = 512,
                 patch_size: int = 16, num_channels: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.num_positions = self.num_patches

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
            bias=False,
        )
        self.position_embedding = nn.Embedding(self.num_positions, hidden_size)

    def forward(self, pixel_values: torch.Tensor,
                patch_attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if patch_attention_mask is None:
            # All patches attended — use sequential position IDs
            position_ids = torch.arange(
                self.num_patches, device=pixel_values.device
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            max_nb_patches_h = max_im_h // self.patch_size
            max_nb_patches_w = max_im_w // self.patch_size
            boundaries = torch.arange(
                1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side,
                device=pixel_values.device,
            )
            position_ids = torch.full(
                size=(batch_size, max_nb_patches_h * max_nb_patches_w),
                fill_value=0,
                device=pixel_values.device,
            )
            for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

                h_indices = torch.arange(nb_patches_h, device=position_ids.device,
                                         dtype=pixel_values.dtype)
                w_indices = torch.arange(nb_patches_w, device=position_ids.device,
                                         dtype=pixel_values.dtype)

                fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
                fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
                position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class NeuronSmolVLMVisionEncoder(nn.Module):
    """
    Full SigLIP-style ViT encoder.
    Input:  pixel_values [B, 3, 512, 512] bf16
    Output: last_hidden_state [B, 1024, 768] bf16
    """

    def __init__(self,
                 hidden_size: int = 768,
                 image_size: int = 512,
                 patch_size: int = 16,
                 num_channels: int = 3,
                 num_attention_heads: int = 12,
                 num_hidden_layers: int = 27,
                 intermediate_size: int = 3072,
                 layer_norm_eps: float = 1e-6,
                 **kwargs):
        super().__init__()

        self.embeddings = NeuronSmolVLMVisionEmbeddings(
            hidden_size=hidden_size,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
        )
        self.encoder = nn.ModuleList([
            NeuronSmolVLMVisionEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)

        for layer in self.encoder:
            hidden_states = layer(hidden_states)

        hidden_states = self.post_layernorm(hidden_states).to(dtype=pixel_values.dtype)
        return hidden_states
