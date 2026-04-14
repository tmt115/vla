"""
Deliverable 4: Text decoder CPU block unit tests.
Tests text GQA attention, SwiGLU MLP, RMSNorm, and decoder layer.

Run:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  cd ~/qwen-port && python -m pytest tests/test_text_blocks.py -v
"""

import types
import sys
import pytest
import torch

NXDI_ROOT = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages"
if NXDI_ROOT not in sys.path:
    sys.path.insert(0, NXDI_ROOT)

# ── cpu_mode patch ─────────────────────────────────────────────────────────────
# On trn1, cpu_mode() returns False (Neuron HW present), so get_rmsnorm_cls()
# returns CustomRMSNorm which requires XLA tensors and fails on CPU tensors.
# Patch it to True so LlamaRMSNorm is used for all CPU block tests.
from unittest.mock import patch as _mock_patch
import pytest

@pytest.fixture(autouse=True)
def _force_cpu_mode(monkeypatch):
    """Force cpu_mode()=True so LlamaRMSNorm is selected instead of CustomRMSNorm."""
    monkeypatch.setattr(
        "neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text.cpu_mode",
        lambda: True,
    )
    # Also patch in the modules.custom_calls path used by attention_base
    try:
        monkeypatch.setattr(
            "neuronx_distributed_inference.modules.custom_calls.cpu_mode",
            lambda: True,
        )
    except AttributeError:
        pass


# ── helpers ────────────────────────────────────────────────────────────────────

def make_text_nc(tp_degree=1, dtype="float32"):
    from neuronx_distributed_inference.models.config import NeuronConfig
    nc = NeuronConfig(tp_degree=tp_degree, torch_dtype=dtype)
    nc.rpl_reduce_dtype = None
    nc.sequence_parallel_enabled = False
    nc.fused_qkv = False
    return nc


def make_text_cfg(hidden_size=128, num_attn_heads=8, num_kv_heads=2,
                   intermediate_size=256, num_layers=2):
    """SimpleNamespace text config for block-level tests."""
    cfg = types.SimpleNamespace()
    cfg.hidden_size = hidden_size
    cfg.num_attention_heads = num_attn_heads
    cfg.num_key_value_heads = num_kv_heads
    cfg.head_dim = hidden_size // num_attn_heads
    cfg.intermediate_size = intermediate_size
    cfg.num_hidden_layers = num_layers
    cfg.hidden_act = "silu"
    cfg.rms_norm_eps = 1e-6
    cfg.rope_theta = 10000.0
    cfg.max_position_embeddings = 512
    cfg.rope_scaling = {"rope_type": "default", "mrope_section": [4, 4, 4]}
    cfg.attention_chunk_size = None
    cfg.sliding_window = None
    cfg.num_cores_per_group = 1
    cfg.neuron_config = make_text_nc()
    return cfg


def make_hf_text_cfg(hidden_size=128, num_attn_heads=8, num_kv_heads=2,
                      intermediate_size=256, num_layers=2):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    tc = Qwen3VLTextConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attn_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        head_dim=hidden_size // num_attn_heads,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 4, 4]},
        max_position_embeddings=512,
    )
    tc._attn_implementation = "eager"
    return tc


# ── Test 1: Text MLP parity ────────────────────────────────────────────────────

class TestTextMLPParity:
    """NeuronLlamaMLP output must match Qwen3VLTextMLP for identical weights."""

    def test_shapes(self):
        from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
        cfg = make_text_cfg(hidden_size=128, intermediate_size=256)
        mlp = NeuronLlamaMLP(cfg)
        mlp.eval()
        x = torch.randn(1, 8, 128)
        out = mlp(x)
        assert out[0].shape == (1, 8, 128), f"Expected (1, 8, 128), got {out[0].shape}"

    def test_parity_with_hf(self):
        """NxDI MLP (SwiGLU) matches HF Qwen3VLTextMLP for same weights."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP
        from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP

        tc = make_hf_text_cfg(hidden_size=128, intermediate_size=256)
        cfg = make_text_cfg(hidden_size=128, intermediate_size=256)

        torch.manual_seed(1)
        hf_mlp = Qwen3VLTextMLP(tc)
        hf_mlp.eval()

        nxdi_mlp = NeuronLlamaMLP(cfg)
        nxdi_mlp.eval()

        # HF and NxDI MLP both use gate_proj / up_proj / down_proj
        nxdi_mlp.load_state_dict(hf_mlp.state_dict(), strict=True)

        x = torch.randn(1, 6, 128)
        with torch.no_grad():
            hf_out = hf_mlp(x)
            nxdi_out = nxdi_mlp(x)[0]

        assert torch.allclose(hf_out, nxdi_out, atol=1e-5), (
            f"MLP parity failed. Max diff: {(hf_out - nxdi_out).abs().max().item():.3e}"
        )

    def test_gate_proj_activation(self):
        """SwiGLU gate: output = gate_proj(x) * silu(up_proj(x))."""
        from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
        cfg = make_text_cfg(hidden_size=64, intermediate_size=128)
        mlp = NeuronLlamaMLP(cfg)
        mlp.eval()

        x = torch.zeros(1, 1, 64)
        with torch.no_grad():
            out, _ = mlp(x)
        # zero input → zero output (silu(0) = 0, gate(0) = 0)
        assert out.abs().max().item() < 1e-6


# ── Test 2: RMSNorm parity ─────────────────────────────────────────────────────

class TestRMSNormParity:
    """LlamaRMSNorm (CPU fallback) matches Qwen3VLTextRMSNorm."""

    def test_parity_with_hf(self):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        hidden_size = 64
        eps = 1e-6
        torch.manual_seed(2)

        hf_norm = Qwen3VLTextRMSNorm(hidden_size, eps=eps)
        nxdi_norm = LlamaRMSNorm(hidden_size, eps=eps)

        # Copy weight
        nxdi_norm.weight.data = hf_norm.weight.data.clone()

        x = torch.randn(2, 8, hidden_size)
        with torch.no_grad():
            hf_out = hf_norm(x)
            nxdi_out = nxdi_norm(x)

        assert torch.allclose(hf_out, nxdi_out, atol=1e-6), (
            f"RMSNorm parity failed. Max diff: {(hf_out - nxdi_out).abs().max().item():.3e}"
        )

    def test_unit_scale(self):
        """With weight=1, RMSNorm should normalize to unit RMS."""
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        hidden_size = 32
        norm = LlamaRMSNorm(hidden_size, eps=1e-6)
        norm.weight.data.fill_(1.0)
        x = torch.randn(4, 8, hidden_size)
        with torch.no_grad():
            out = norm(x)
        # Each token's RMS should be ≈ 1.0
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), (
            f"RMS of normalised output != 1. Got: {rms.mean():.4f}"
        )


# ── Test 3: Text attention shapes and numerics ─────────────────────────────────

class TestTextAttention:
    """NeuronQwen3VLAttention (GQA) shapes and basic numerics."""

    def _make_attn(self, hidden=128, n_heads=8, n_kv=2):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
            NeuronQwen3VLAttention,
        )
        cfg = make_text_cfg(hidden_size=hidden, num_attn_heads=n_heads, num_kv_heads=n_kv)
        return NeuronQwen3VLAttention(cfg), cfg

    def test_output_shape(self):
        attn, cfg = self._make_attn()
        attn.eval()
        bs, seq = 1, 12
        x = torch.randn(bs, seq, cfg.hidden_size)
        pos_ids = torch.arange(seq).unsqueeze(0)

        with torch.no_grad():
            out = attn(x, position_ids=pos_ids)

        assert out.hidden_states.shape == (bs, seq, cfg.hidden_size), (
            f"Expected ({bs}, {seq}, {cfg.hidden_size}), got {out.hidden_states.shape}"
        )

    def test_state_dict_keys(self):
        """Verify q/k/v are stored as separate linear layers (fused_qkv=False)."""
        attn, _ = self._make_attn()
        keys = set(attn.state_dict().keys())
        # Separate QKV keys
        assert "qkv_proj.q_proj.weight" in keys
        assert "qkv_proj.k_proj.weight" in keys
        assert "qkv_proj.v_proj.weight" in keys
        # Per-head layer norms
        assert "q_layernorm.weight" in keys
        assert "k_layernorm.weight" in keys

    def test_output_finite(self):
        attn, cfg = self._make_attn()
        attn.eval()
        x = torch.randn(1, 10, cfg.hidden_size)
        pos_ids = torch.arange(10).unsqueeze(0)
        with torch.no_grad():
            out = attn(x, position_ids=pos_ids)
        assert torch.isfinite(out.hidden_states).all(), (
            "Text attention output contains NaN/Inf"
        )

    def test_mrope_3d_position_ids(self):
        """3D MRoPE position ids [3, B, S] are accepted without error."""
        attn, cfg = self._make_attn()
        attn.eval()
        bs, seq = 1, 8
        x = torch.randn(bs, seq, cfg.hidden_size)
        # 3D rotary position ids: [t, h, w] all identical (text-only)
        rot_ids = torch.stack([
            torch.arange(seq).unsqueeze(0),   # t
            torch.arange(seq).unsqueeze(0),   # h
            torch.arange(seq).unsqueeze(0),   # w
        ])  # [3, 1, 8]
        with torch.no_grad():
            out = attn(x, rotary_position_ids=rot_ids)
        assert out.hidden_states.shape == (bs, seq, cfg.hidden_size)
        assert torch.isfinite(out.hidden_states).all()


# ── Test 4: Decoder layer ──────────────────────────────────────────────────────

class TestDecoderLayer:
    """NeuronQwen3VLDecoderLayer shapes and basic properties."""

    def _make_layer(self, hidden=128, n_heads=8, n_kv=2, intermediate=256):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
            NeuronQwen3VLDecoderLayer,
        )
        cfg = make_text_cfg(hidden_size=hidden, num_attn_heads=n_heads,
                             num_kv_heads=n_kv, intermediate_size=intermediate)
        return NeuronQwen3VLDecoderLayer(cfg), cfg

    def test_output_shape(self):
        layer, cfg = self._make_layer()
        layer.eval()
        bs, seq = 1, 8
        x = torch.randn(bs, seq, cfg.hidden_size)
        pos_ids = torch.arange(seq).unsqueeze(0)

        with torch.no_grad():
            out = layer(x, position_ids=pos_ids)

        hidden_states = out[0]
        assert hidden_states.shape == (bs, seq, cfg.hidden_size), (
            f"Expected ({bs}, {seq}, {cfg.hidden_size}), got {hidden_states.shape}"
        )

    def test_output_finite(self):
        layer, cfg = self._make_layer()
        layer.eval()
        x = torch.randn(1, 6, cfg.hidden_size)
        pos_ids = torch.arange(6).unsqueeze(0)
        with torch.no_grad():
            out = layer(x, position_ids=pos_ids)
        assert torch.isfinite(out[0]).all(), "Decoder layer output contains NaN/Inf"

    def test_residual_stream_shape(self):
        """Output hidden dim must match input (residual connection)."""
        layer, cfg = self._make_layer(hidden=64, n_heads=4, n_kv=2, intermediate=128)
        layer.eval()
        x = torch.randn(1, 5, 64)
        pos_ids = torch.arange(5).unsqueeze(0)
        with torch.no_grad():
            out = layer(x, position_ids=pos_ids)
        assert out[0].shape == x.shape, "Residual stream shape must match input"

    def test_state_dict_structure(self):
        """Decoder layer must have expected sub-module keys."""
        layer, _ = self._make_layer()
        keys = set(layer.state_dict().keys())
        # Attention
        assert any("self_attn" in k for k in keys), "Missing self_attn keys"
        # MLP
        assert any("mlp" in k for k in keys), "Missing mlp keys"
        # Norms
        assert any("input_layernorm" in k for k in keys), "Missing input_layernorm keys"
        assert any("post_attention_layernorm" in k for k in keys), "Missing post_attn_ln keys"

    def test_weight_loading_from_hf(self):
        """Weights loaded from HF Qwen3VLTextDecoderLayer produce correct output shape."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
            NeuronQwen3VLDecoderLayer,
        )
        tc = make_hf_text_cfg(hidden_size=128, num_attn_heads=8, num_kv_heads=2,
                               intermediate_size=256)
        cfg = make_text_cfg(hidden_size=128, num_attn_heads=8, num_kv_heads=2,
                             intermediate_size=256)

        torch.manual_seed(42)
        hf_layer = Qwen3VLTextDecoderLayer(tc, layer_idx=0)
        hf_layer.eval()

        nxdi_layer = NeuronQwen3VLDecoderLayer(cfg)
        nxdi_layer.eval()

        # Build converted state dict following convert_hf_to_neuron_state_dict logic
        hf_sd = hf_layer.state_dict()
        converted = {}
        attn_renames = {
            "self_attn.q_proj.": "self_attn.qkv_proj.q_proj.",
            "self_attn.k_proj.": "self_attn.qkv_proj.k_proj.",
            "self_attn.v_proj.": "self_attn.qkv_proj.v_proj.",
            "self_attn.o_proj.": "self_attn.o_proj.o_proj.",
            "self_attn.q_norm.": "self_attn.q_layernorm.",
            "self_attn.k_norm.": "self_attn.k_layernorm.",
        }
        for k, v in hf_sd.items():
            new_k = k
            for old_prefix, new_prefix in attn_renames.items():
                if old_prefix in new_k:
                    new_k = new_k.replace(old_prefix, new_prefix)
                    break
            converted[new_k] = v.clone()

        # rank_util.rank is a runtime buffer (SPMDRank) added when parallel state is
        # initialized — it is NOT a checkpoint key and must be excluded from strict load.
        missing, unexpected = nxdi_layer.load_state_dict(converted, strict=False)
        non_rank_missing = [k for k in missing if "rank_util" not in k]
        assert not non_rank_missing, f"Unexpected missing keys: {non_rank_missing}"
        assert not unexpected, f"Unexpected extra keys: {unexpected}"

        # Now run forward and verify shape
        x = torch.randn(1, 6, 128)
        pos_ids = torch.arange(6).unsqueeze(0)
        with torch.no_grad():
            out = nxdi_layer(x, position_ids=pos_ids)
        assert out[0].shape == (1, 6, 128)
        assert torch.isfinite(out[0]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
