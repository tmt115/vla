"""
compile_vit.py — Attempt to compile Qwen3-VL ViT to Neuron NEFF.

Static assumptions:
  grid_thw = [[1, 16, 16]] -> NPATCH=256 patches -> 64 merged tokens (after spatial_merge_size=2)
  cu_seqlens = [0, 256] (all 256 patches in one sequence, no split needed)

Dynamic op fixes:
  1. rot_pos_emb: replaced with pre-computed static buffer
  2. fast_pos_embed_interpolate: replaced with pre-computed static buffer
  3. Attention cu_seqlens split: for a single image (cu_seqlens=[0,256]), lengths=[256]
     -> torch.split with [256] means no actual split, just identity.
     We monkey-patch the attention forward to do standard full attention without split.
"""

import sys
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
from types import MethodType

sys.path.insert(0, '/home/ubuntu/groot-n1-port')
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
from safetensors.torch import load_file
from config_constants import (
    VIT_PATCH_SIZE, VIT_TEMPORAL_PATCH_SIZE, VIT_SPATIAL_MERGE_SIZE,
    LLM_ROPE_THETA, LLM_TOTAL_LAYERS, MODEL_PATH, VISION_TOKENS_PER_IMAGE,
)

NPATCH = 256       # T=1, H=16, W=16 patches before merger
N_MERGED = VISION_TOKENS_PER_IMAGE  # 64 tokens after merger
GRID_THW = torch.tensor([[1, 16, 16]], dtype=torch.long)
SAVE_PATH = '/home/ubuntu/groot-n1-port/compiled/vit/model.pt'


def _patch_attn_forward(attn_module):
    """
    Monkey-patch Qwen3VLVisionAttention.forward to avoid dynamic cu_seqlens split.

    For a single image with NPATCH=256, cu_seqlens=[0,256] -> lengths=[256].
    torch.split(tensor, [256]) is a no-op (returns 1-tuple with the full tensor).
    We replace the entire attention with standard scaled dot-product attention
    over the full sequence -- numerically identical for a single image.
    """
    scaling = attn_module.scaling

    def static_attn_forward(self, hidden_states, cu_seqlens=None,
                            rotary_pos_emb=None, position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        if position_embeddings is not None:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Standard full-sequence attention: [1, num_heads, seq, head_dim]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

    attn_module.forward = MethodType(static_attn_forward, attn_module)


print("Loading checkpoint...")
hf_sd = {}
for f in sorted(glob.glob(os.path.join(MODEL_PATH, '*.safetensors'))):
    hf_sd.update(load_file(f))
print(f"  {len(hf_sd)} tensors")

print("Building Qwen3VL model...")
qwen_cfg = Qwen3VLConfig(
    text_config=dict(
        hidden_size=2048, num_hidden_layers=LLM_TOTAL_LAYERS,
        num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, intermediate_size=6144,
        vocab_size=151936, max_position_embeddings=32768,
        rope_theta=LLM_ROPE_THETA, hidden_act='silu',
        rope_scaling={'rope_type': 'default', 'mrope_section': [24, 20, 20]},
    ),
    vision_config=dict(
        depth=24, hidden_size=1024, hidden_act='quick_gelu',
        intermediate_size=4096, num_heads=16, in_channels=3,
        patch_size=VIT_PATCH_SIZE, spatial_merge_size=VIT_SPATIAL_MERGE_SIZE,
        temporal_patch_size=VIT_TEMPORAL_PATCH_SIZE,
        out_hidden_size=2048, num_position_embeddings=2304,
        deepstack_visual_indexes=[5, 11, 17],
    ),
    image_token_id=151655,
)
qwen_cfg._attn_implementation = 'eager'
model = Qwen3VLForConditionalGeneration(qwen_cfg).bfloat16().eval()

pfx = 'backbone.model.'
backbone_sd = {k[len(pfx):]: v for k, v in hf_sd.items() if k.startswith(pfx)}
model.load_state_dict(backbone_sd, strict=False)
print("  Backbone weights loaded.")

vit = model.model.visual

# Patch all attention blocks to use static full attention
for blk in vit.blocks:
    _patch_attn_forward(blk.attn)

# Pre-compute static rotary and pos embeddings for grid_thw=[1,16,16]
with torch.no_grad():
    static_rotary = vit.rot_pos_emb(GRID_THW)
    static_pos_embed = vit.fast_pos_embed_interpolate(GRID_THW)

print(f"Static rotary: {static_rotary.shape}")
print(f"Static pos_embed: {static_pos_embed.shape}")


class StaticViTWrapper(nn.Module):
    """Fully static ViT wrapper for Neuron tracing.

    Input:  pixel_values [NPATCH, 3*temporal*patch*patch] BF16
    Output: merged_tokens [N_MERGED, 2048] BF16
    """

    def __init__(self, vit, static_rotary, static_pos_embed):
        super().__init__()
        self.vit = vit
        self.register_buffer('_rotary', static_rotary.bfloat16())
        self.register_buffer('_pos_embed', static_pos_embed.bfloat16())

    def forward(self, pixel_values):
        vit = self.vit
        hidden_states = vit.patch_embed(pixel_values)
        hidden_states = hidden_states + self._pos_embed

        seq_len = hidden_states.size(0)
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary = self._rotary.reshape(seq_len, -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # cu_seqlens not used by patched static_attn_forward
        cu_seqlens = torch.zeros(2, dtype=torch.int32)

        for blk in vit.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        hidden_states = vit.merger(hidden_states)
        return hidden_states


wrapper = StaticViTWrapper(vit, static_rotary, static_pos_embed).bfloat16().eval()

pixel_values = torch.randn(
    NPATCH, 3 * VIT_TEMPORAL_PATCH_SIZE * VIT_PATCH_SIZE * VIT_PATCH_SIZE,
    dtype=torch.bfloat16,
)

print("\nTesting CPU forward pass...")
with torch.no_grad():
    cpu_out = wrapper(pixel_values)
print(f"CPU output: {cpu_out.shape}, dtype={cpu_out.dtype}")
assert cpu_out.shape == (N_MERGED, 2048), f"Expected ({N_MERGED}, 2048) got {cpu_out.shape}"
print("CPU forward PASSED.")

print("\nAttempting torch_neuronx.trace...")
print("Compiler args: --auto-cast=matmult --optlevel 3 --model-type=unet-inference")

try:
    traced = torch_neuronx.trace(
        wrapper,
        (pixel_values,),
        compiler_args="--auto-cast=matmult --optlevel 3 --model-type=unet-inference",
    )
    out_neuron = traced(pixel_values)
    print(f"Trace SUCCESS! Output: {out_neuron.shape}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.jit.save(traced, SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")

    cpu_f = cpu_out.float().flatten()
    nef_f = out_neuron.float().flatten()
    cos_sim = F.cosine_similarity(cpu_f.unsqueeze(0), nef_f.unsqueeze(0)).item()
    print(f"Neuron vs CPU cos_sim: {cos_sim:.6f}")

except Exception as e:
    import traceback
    print(f"\nTrace FAILED: {type(e).__name__}: {e}")
    print("Full traceback:")
    traceback.print_exc()
