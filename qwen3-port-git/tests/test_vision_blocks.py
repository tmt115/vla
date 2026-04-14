"""
Deliverable 1: Vision encoder CPU block unit tests.
Tests vision attention, vision MLP, MRoPE position-id builder, and vision block.

Run:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  cd ~/qwen-port && python -m pytest tests/test_vision_blocks.py -v
"""

import types
import sys
import pytest
import torch
import torch.nn as nn

# ── helpers ────────────────────────────────────────────────────────────────────

NXDI_ROOT = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages"
if NXDI_ROOT not in sys.path:
    sys.path.insert(0, NXDI_ROOT)


def make_vision_nc(tp_degree=1, dtype="float32"):
    """Minimal NeuronConfig for vision blocks."""
    from neuronx_distributed_inference.models.config import NeuronConfig
    nc = NeuronConfig(tp_degree=tp_degree, torch_dtype=dtype)
    nc.rpl_reduce_dtype = None
    nc.sequence_parallel_enabled = False
    nc.fused_qkv = False
    return nc


def make_vision_cfg(hidden_size=64, num_heads=4, intermediate_size=128):
    """SimpleNamespace config for vision block tests."""
    cfg = types.SimpleNamespace()
    cfg.hidden_size = hidden_size
    cfg.num_heads = num_heads
    cfg.intermediate_size = intermediate_size
    cfg.hidden_act = "gelu_pytorch_tanh"
    cfg.neuron_config = make_vision_nc()
    return cfg


def make_hf_vision_cfg(hidden_size=64, num_heads=4, intermediate_size=128,
                        depth=2, patch_size=4, temporal_patch_size=2,
                        spatial_merge_size=2, out_hidden_size=32,
                        num_position_embeddings=16):
    """transformers Qwen3VLVisionConfig for HF reference tests."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    vc = Qwen3VLVisionConfig(
        depth=depth,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        in_channels=3,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        out_hidden_size=out_hidden_size,
        num_position_embeddings=num_position_embeddings,
        hidden_act="gelu_pytorch_tanh",
    )
    vc._attn_implementation = "eager"
    return vc


# ── Test 1: Vision MLP parity ──────────────────────────────────────────────────

class TestVisionMLPParity:
    """NeuronQwen3VLVisionMLP output must match Qwen3VLVisionMLP for identical weights."""

    def test_shapes(self):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionMLP,
        )
        cfg = make_vision_cfg(hidden_size=64, intermediate_size=128)
        mlp = NeuronQwen3VLVisionMLP(cfg)
        x = torch.randn(12, 64)  # (seq_len, hidden)
        out = mlp(x)
        assert out.shape == (12, 64), f"Expected (12, 64), got {out.shape}"

    def test_parity_with_hf(self):
        """NxDI MLP produces the same output as HF MLP for identical weights."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionMLP
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionMLP,
        )
        vc = make_hf_vision_cfg(hidden_size=64, intermediate_size=128)
        cfg = make_vision_cfg(hidden_size=64, intermediate_size=128)

        torch.manual_seed(0)
        hf_mlp = Qwen3VLVisionMLP(vc)
        hf_mlp.eval()

        nxdi_mlp = NeuronQwen3VLVisionMLP(cfg)
        nxdi_mlp.eval()

        # HF uses linear_fc1 / linear_fc2 — same key names as NxDI (confirmed from state dicts)
        nxdi_mlp.load_state_dict(hf_mlp.state_dict(), strict=True)

        x = torch.randn(8, 64)
        with torch.no_grad():
            hf_out = hf_mlp(x)
            nxdi_out = nxdi_mlp(x)

        assert torch.allclose(hf_out, nxdi_out, atol=1e-5), (
            f"MLP parity failed. Max diff: {(hf_out - nxdi_out).abs().max().item():.3e}"
        )


# ── Test 2: Vision attention shapes ────────────────────────────────────────────

class TestVisionAttention:
    """NeuronQwen3VLVisionAttention forward produces correct shapes."""

    def _make_attn(self, hidden_size=64, num_heads=4):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionAttention,
        )
        cfg = make_vision_cfg(hidden_size=hidden_size, num_heads=num_heads)
        return NeuronQwen3VLVisionAttention(cfg), cfg

    def test_forward_shape(self):
        attn, cfg = self._make_attn(hidden_size=64, num_heads=4)
        attn.eval()
        seq_len = 16
        hidden = cfg.hidden_size
        head_dim = hidden // cfg.num_heads  # 16

        x = torch.randn(1, seq_len, hidden)
        cos = torch.randn(1, seq_len, head_dim)
        sin = torch.randn(1, seq_len, head_dim)

        with torch.no_grad():
            out = attn(x, position_embeddings=(cos, sin))

        assert out.hidden_states.shape == (1, seq_len, hidden), (
            f"Expected (1, {seq_len}, {hidden}), got {out.hidden_states.shape}"
        )

    def test_forward_with_attention_mask(self):
        """2D boolean attention mask should gate cross-image attention."""
        attn, cfg = self._make_attn(hidden_size=64, num_heads=4)
        attn.eval()
        seq_len = 16
        hidden = cfg.hidden_size
        head_dim = hidden // cfg.num_heads

        x = torch.randn(1, seq_len, hidden)
        cos = torch.randn(1, seq_len, head_dim)
        sin = torch.randn(1, seq_len, head_dim)
        # Block-diagonal mask: 2 images of 8 tokens each
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        mask[:8, :8] = True
        mask[8:, 8:] = True

        with torch.no_grad():
            out = attn(x, position_embeddings=(cos, sin), attention_mask=mask)

        assert out.hidden_states.shape == (1, seq_len, hidden)

    def test_output_is_finite(self):
        attn, cfg = self._make_attn()
        attn.eval()
        x = torch.randn(1, 12, cfg.hidden_size)
        head_dim = cfg.hidden_size // cfg.num_heads
        cos = torch.randn(1, 12, head_dim)
        sin = torch.randn(1, 12, head_dim)
        with torch.no_grad():
            out = attn(x, position_embeddings=(cos, sin))
        assert torch.isfinite(out.hidden_states).all(), "Vision attention output contains NaN/Inf"


# ── Test 3: MRoPE position ID builder ─────────────────────────────────────────

class TestMRoPEPositionIDs:
    """Validate vision position id creation and rotary positional embedding builder."""

    def test_create_vision_position_ids_single_image(self):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionModelWrapper, VISION_POSITION_ID_PAD_VALUE,
        )
        # Single image: T=1, H=4, W=4 → 16 tokens
        grid_thw = torch.tensor([[1, 4, 4]])
        ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(grid_thw)
        assert ids.shape == (16,), f"Expected (16,), got {ids.shape}"
        assert (ids == 0).all(), "All tokens should belong to image 0"

    def test_create_vision_position_ids_two_images(self):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionModelWrapper,
        )
        # Image 0: 2×2 → 4 tokens; Image 1: 2×3 → 6 tokens
        grid_thw = torch.tensor([[1, 2, 2], [1, 2, 3]])
        ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(grid_thw)
        assert ids.shape == (10,), f"Expected (10,), got {ids.shape}"
        assert (ids[:4] == 0).all(), "First 4 tokens: image 0"
        assert (ids[4:] == 1).all(), "Next 6 tokens: image 1"

    def test_vision_attention_mask_from_pos_ids(self):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionModel,
        )
        # 4 tokens: 2 from image 0, 2 from image 1
        pos_ids = torch.tensor([0, 0, 1, 1])
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        assert mask.shape == (4, 4)
        # Block diagonal: tokens 0-1 attend to 0-1; tokens 2-3 attend to 2-3
        expected = torch.tensor([
            [True,  True,  False, False],
            [True,  True,  False, False],
            [False, False, True,  True ],
            [False, False, True,  True ],
        ])
        assert torch.equal(mask, expected), f"Attention mask mismatch:\n{mask}"

    def test_vision_attention_mask_with_padding(self):
        """Padded tokens (id=-1) must be masked out."""
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionModel, VISION_POSITION_ID_PAD_VALUE,
        )
        pos_ids = torch.tensor([0, 0, VISION_POSITION_ID_PAD_VALUE, VISION_POSITION_ID_PAD_VALUE])
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        # Padded tokens should not attend to anything
        assert mask[2, :].sum() == 0, "Padded token should not attend to anything"
        assert mask[:, 2].sum() == 0, "Nothing should attend to padded token"

    def test_mrope_text_rotation_structure(self):
        """NeuronQwen3VLRotaryEmbedding: text-only (2D) position_ids expand correctly."""
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
            NeuronQwen3VLRotaryEmbedding,
        )
        cfg = types.SimpleNamespace()
        cfg.max_position_embeddings = 32
        cfg.rope_theta = 10000.0
        cfg.hidden_size = 128
        cfg.num_attention_heads = 4
        cfg.rope_scaling = {"rope_type": "default", "mrope_section": [4, 4, 4]}

        rope = NeuronQwen3VLRotaryEmbedding(cfg)
        # Text-only: 2D position_ids [batch, seq_len]
        pos_ids = torch.arange(8).unsqueeze(0)  # [1, 8]
        x = torch.randn(1, 8, 128)
        cos, sin = rope(x, pos_ids)
        # RoPE returns [bs, seq, head_dim] where head_dim = hidden_size // num_attention_heads = 32
        head_dim = cfg.hidden_size // cfg.num_attention_heads  # 32
        assert cos.shape == sin.shape
        assert cos.shape[-1] == head_dim, (
            f"Expected last dim={head_dim} (head_dim), got {cos.shape}"
        )
        assert torch.isfinite(cos).all()
        assert torch.isfinite(sin).all()

    def test_mrope_3d_position_ids(self):
        """3D MRoPE position_ids [3, B, S] produce correct output shape."""
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
            NeuronQwen3VLRotaryEmbedding,
        )
        cfg = types.SimpleNamespace()
        cfg.max_position_embeddings = 64
        cfg.rope_theta = 10000.0
        cfg.hidden_size = 128
        cfg.num_attention_heads = 4
        cfg.rope_scaling = {"rope_type": "default", "mrope_section": [4, 4, 4]}

        rope = NeuronQwen3VLRotaryEmbedding(cfg)
        # 3D MRoPE: [3, batch, seq_len]
        pos_ids = torch.stack([
            torch.arange(8).unsqueeze(0),   # temporal (t)
            torch.arange(8).unsqueeze(0),   # height (h)
            torch.arange(8).unsqueeze(0),   # width (w)
        ])  # shape [3, 1, 8]
        x = torch.randn(1, 8, 128)
        cos, sin = rope(x, pos_ids)
        # head_dim = hidden_size // num_attention_heads = 128 // 4 = 32
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        assert cos.shape[-1] == head_dim, (
            f"Expected last dim={head_dim} (head_dim), got {cos.shape}"
        )
        assert torch.isfinite(cos).all()
        assert torch.isfinite(sin).all()


# ── Test 4: Vision block shapes ────────────────────────────────────────────────

class TestVisionBlock:
    """NeuronQwen3VLVisionBlock forward produces correct shapes."""

    def test_forward_shape(self):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionBlock,
        )
        cfg = make_vision_cfg(hidden_size=64, num_heads=4)
        block = NeuronQwen3VLVisionBlock(cfg)
        block.eval()

        seq_len = 16
        hidden = 64
        head_dim = hidden // 4

        x = torch.randn(1, seq_len, hidden)
        cos = torch.randn(1, seq_len, head_dim)
        sin = torch.randn(1, seq_len, head_dim)

        with torch.no_grad():
            out = block(x, position_embeddings=(cos, sin))

        assert out.shape == (1, seq_len, hidden), (
            f"Expected (1, {seq_len}, {hidden}), got {out.shape}"
        )

    def test_output_is_finite(self):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionBlock,
        )
        cfg = make_vision_cfg(hidden_size=64, num_heads=4)
        block = NeuronQwen3VLVisionBlock(cfg)
        block.eval()

        x = torch.randn(1, 9, 64)
        head_dim = 64 // 4
        cos = torch.randn(1, 9, head_dim)
        sin = torch.randn(1, 9, head_dim)

        with torch.no_grad():
            out = block(x, position_embeddings=(cos, sin))

        assert torch.isfinite(out).all(), "Vision block output contains NaN/Inf"

    def test_residual_stream(self):
        """Input and output hidden dim must match (residual stream preserved)."""
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionBlock,
        )
        cfg = make_vision_cfg(hidden_size=128, num_heads=8)
        block = NeuronQwen3VLVisionBlock(cfg)
        x = torch.randn(1, 8, 128)
        head_dim = 128 // 8
        cos = torch.randn(1, 8, head_dim)
        sin = torch.randn(1, 8, head_dim)
        with torch.no_grad():
            out = block(x, position_embeddings=(cos, sin))
        assert out.shape == x.shape, "Residual stream shape must match input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
