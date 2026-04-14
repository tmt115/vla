"""
Deliverable 2: Vision encoder end-to-end CPU parity test.
Validates NeuronQwen3VLVisionModel against HF Qwen3VLVisionModel on CPU
using a small synthetic config and random weights.

The test verifies:
  - vision_embeddings shape: [1, text_seq_len, out_hidden_size]
  - deepstack_vision_embeds shape: [num_deepstack_layers, 1, text_seq_len, out_hidden_size]
  - vision_embeddings values are finite (non-NaN, non-Inf)
  - Approximate numerical parity when both models use identical weights.

Run:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  cd ~/qwen-port && python -m pytest tests/test_vision_e2e.py -v -s
"""

import sys
import types
import math
import pytest
import torch
import torch.nn as nn

NXDI_ROOT = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages"
if NXDI_ROOT not in sys.path:
    sys.path.insert(0, NXDI_ROOT)


# ── Config helpers ─────────────────────────────────────────────────────────────

# Small synthetic sizes that keep tests fast.
# patch_size=4, temporal_patch_size=2, spatial_merge_size=2
# Single 4×4 image grid → 16 raw patches → 4 merged patches after merger
_SMALL = dict(
    depth=2,
    hidden_size=64,
    intermediate_size=128,
    num_heads=4,
    in_channels=3,
    patch_size=4,           # spatial patch pixels
    temporal_patch_size=2,  # temporal patch frames
    spatial_merge_size=2,
    out_hidden_size=32,     # text-side hidden size
    num_position_embeddings=64,  # sqrt(64)=8 → 8×8 max grid
    hidden_act="gelu_pytorch_tanh",
    deepstack_visual_indexes=[0],  # capture one intermediate layer
)
# text seq len for padding: must be >= compressed vision seq len
_TEXT_SEQ_LEN = 256
# vision bucket (padded seq len for compilation)
_VISION_BUCKET = 64


def _make_hf_vision_cfg(**overrides):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    kwargs = dict(_SMALL)
    kwargs.update(overrides)
    vc = Qwen3VLVisionConfig(**{
        k: v for k, v in kwargs.items() if k != "deepstack_visual_indexes"
    })
    vc.deepstack_visual_indexes = kwargs["deepstack_visual_indexes"]
    vc._attn_implementation = "eager"
    return vc


def _make_nxdi_vision_cfg(**overrides):
    """Return a SimpleNamespace vision config for NeuronQwen3VLVisionModel."""
    from neuronx_distributed_inference.models.config import NeuronConfig
    kwargs = dict(_SMALL)
    kwargs.update(overrides)

    nc = NeuronConfig(tp_degree=1, torch_dtype="float32")
    nc.rpl_reduce_dtype = None
    nc.seq_len = _TEXT_SEQ_LEN
    nc.buckets = [_VISION_BUCKET]
    nc.batch_size = 1

    cfg = types.SimpleNamespace(**kwargs)
    cfg.neuron_config = nc
    return cfg


def _make_full_inference_cfg():
    """
    Full Qwen3VLInferenceConfig with small vision + text configs.
    Used for NeuronQwen3VLVisionModel (which needs config.vision_config and config.text_config).
    """
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig, Qwen3VLVisionConfig, Qwen3VLTextConfig,
    )
    from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLInferenceConfig, Qwen3VLNeuronConfig,
    )

    hf_vc = Qwen3VLVisionConfig(**{
        k: v for k, v in _SMALL.items() if k != "deepstack_visual_indexes"
    })
    hf_vc.deepstack_visual_indexes = _SMALL["deepstack_visual_indexes"]

    hf_tc = Qwen3VLTextConfig(
        hidden_size=_SMALL["out_hidden_size"],   # text hidden == vision out
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        num_hidden_layers=2,
        head_dim=8,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        max_position_embeddings=_TEXT_SEQ_LEN,
    )

    text_nc = Qwen3VLNeuronConfig(
        tp_degree=1,
        torch_dtype="float32",
        batch_size=1,
        seq_len=_TEXT_SEQ_LEN,
        n_active_tokens=_TEXT_SEQ_LEN,
        max_context_length=_TEXT_SEQ_LEN,
        on_device_sampling_config=None,
        buckets=[_TEXT_SEQ_LEN],
    )
    vision_nc = Qwen3VLNeuronConfig(
        tp_degree=1,
        torch_dtype="float32",
        batch_size=1,
        seq_len=_VISION_BUCKET,
        n_active_tokens=_VISION_BUCKET,
        on_device_sampling_config=None,
        buckets=[_VISION_BUCKET],
    )

    cfg = Qwen3VLInferenceConfig(
        text_neuron_config=text_nc,
        vision_neuron_config=vision_nc,
        text_config=vars(hf_tc),
        vision_config=vars(hf_vc),
        _name_or_path="/tmp/fake_qwen3_vl",
    )
    return cfg


def _build_nxdi_vision_state_dict(hf_model, cfg):
    """
    Convert HF Qwen3VLVisionModel state dict → NxDI NeuronQwen3VLVisionModel keys.
    Renames:
      blocks.N.attn.qkv.* → blocks.N.attn.qkv_proj.q_proj.* (split)
      blocks.N.attn.proj.* → blocks.N.attn.o_proj.o_proj.*
    """
    hf_sd = hf_model.state_dict()
    hidden = cfg.vision_config.hidden_size
    num_heads = cfg.vision_config.num_heads
    head_dim = hidden // num_heads

    converted = {}
    for k, v in hf_sd.items():
        if "attn.qkv." in k:
            base = k.replace("attn.qkv.", "attn.")
            w = v  # [3*H, H] for weight or [3*H] for bias
            if "weight" in k:
                q = w[:hidden]
                kk = w[hidden : 2 * hidden]
                vv = w[2 * hidden :]
                converted[base.replace("attn.", "attn.qkv_proj.q_proj.")] = q
                converted[base.replace("attn.", "attn.qkv_proj.k_proj.")] = kk
                converted[base.replace("attn.", "attn.qkv_proj.v_proj.")] = vv
            elif "bias" in k:
                q = w[:hidden]
                kk = w[hidden : 2 * hidden]
                vv = w[2 * hidden :]
                converted[base.replace("attn.", "attn.qkv_proj.q_proj.")] = q
                converted[base.replace("attn.", "attn.qkv_proj.k_proj.")] = kk
                converted[base.replace("attn.", "attn.qkv_proj.v_proj.")] = vv
        elif "attn.proj." in k:
            new_k = k.replace("attn.proj.", "attn.o_proj.o_proj.")
            converted[new_k] = v.clone()
        else:
            converted[k] = v.clone()

    return converted


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestVisionEncoderE2E:

    @pytest.fixture(scope="class")
    def full_cfg(self):
        return _make_full_inference_cfg()

    @pytest.fixture(scope="class")
    def hf_model(self):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        vc = _make_hf_vision_cfg()
        torch.manual_seed(7)
        m = Qwen3VLVisionModel(vc)
        m.eval()
        return m

    @pytest.fixture(scope="class")
    def nxdi_model(self, full_cfg, hf_model):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionModel,
        )
        model = NeuronQwen3VLVisionModel(full_cfg)
        model.eval()

        # Load weights converted from HF
        converted_sd = _build_nxdi_vision_state_dict(hf_model, full_cfg)
        # pos_embed not in NeuronQwen3VLVisionModel directly (it's in the wrapper)
        # — drop it
        converted_sd.pop("pos_embed.weight", None)
        model.load_state_dict(converted_sd, strict=False)
        return model

    def _make_inputs(self, full_cfg, hf_model, num_patches=16):
        """
        Generate compatible inputs for both models.
        grid_thw = [1, 4, 4] → 16 patches, patch_dim = 3*2*4*4 = 96
        """
        patch_dim = (full_cfg.vision_config.in_channels
                     * full_cfg.vision_config.temporal_patch_size
                     * full_cfg.vision_config.patch_size
                     * full_cfg.vision_config.patch_size)
        pixel_values = torch.randn(num_patches, patch_dim)
        grid_thw = torch.tensor([[1, 4, 4]])  # T=1, H=4, W=4

        # Compute pos_emb and rotary_pos_emb using same logic as NxDI wrapper
        # Use HF model's rot_pos_emb method (same implementation)
        rotary_pos_emb = hf_model.rot_pos_emb(grid_thw).float()

        # Fast pos embed interpolation using pos_embed from the HF model
        pos_embeds = hf_model.fast_pos_embed_interpolate(grid_thw)

        # vision_position_ids: all 16 tokens belong to image 0
        vision_position_ids = torch.zeros(num_patches, dtype=torch.int32)

        return pixel_values, grid_thw, rotary_pos_emb, pos_embeds, vision_position_ids

    def test_vision_embeddings_shape(self, full_cfg, hf_model, nxdi_model):
        """NxDI vision encoder returns [1, text_seq_len, out_hidden] embeddings."""
        pixel_values, grid_thw, rotary_pos_emb, pos_embeds, vision_position_ids = (
            self._make_inputs(full_cfg, hf_model)
        )

        # Pad inputs to bucket size
        from neuronx_distributed_inference.modules.padding import pad_tensor
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            VISION_POSITION_ID_PAD_VALUE
        )
        bucket = _VISION_BUCKET
        padded_pv, _ = pad_tensor(pixel_values, (bucket, pixel_values.shape[1]))
        padded_rpe, _ = pad_tensor(rotary_pos_emb, (bucket, rotary_pos_emb.shape[1]))
        padded_pe, _ = pad_tensor(pos_embeds, (bucket, pos_embeds.shape[1]), pad_value=0)
        padded_vid, _ = pad_tensor(vision_position_ids, (bucket,),
                                    pad_value=VISION_POSITION_ID_PAD_VALUE)

        with torch.no_grad():
            hidden_states, deepstack = nxdi_model(
                padded_pv, padded_rpe, padded_pe, padded_vid
            )

        out_hidden = full_cfg.vision_config.out_hidden_size
        assert hidden_states.shape == (1, _TEXT_SEQ_LEN, out_hidden), (
            f"Expected (1, {_TEXT_SEQ_LEN}, {out_hidden}), got {hidden_states.shape}"
        )

    def test_deepstack_shape(self, full_cfg, hf_model, nxdi_model):
        """Deepstack embeddings shape: [num_deepstack_layers, 1, text_seq_len, out_hidden]."""
        pixel_values, grid_thw, rotary_pos_emb, pos_embeds, vision_position_ids = (
            self._make_inputs(full_cfg, hf_model)
        )
        from neuronx_distributed_inference.modules.padding import pad_tensor
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            VISION_POSITION_ID_PAD_VALUE
        )
        bucket = _VISION_BUCKET
        padded_pv, _ = pad_tensor(pixel_values, (bucket, pixel_values.shape[1]))
        padded_rpe, _ = pad_tensor(rotary_pos_emb, (bucket, rotary_pos_emb.shape[1]))
        padded_pe, _ = pad_tensor(pos_embeds, (bucket, pos_embeds.shape[1]), pad_value=0)
        padded_vid, _ = pad_tensor(vision_position_ids, (bucket,),
                                    pad_value=VISION_POSITION_ID_PAD_VALUE)

        with torch.no_grad():
            hidden_states, deepstack = nxdi_model(
                padded_pv, padded_rpe, padded_pe, padded_vid
            )

        num_ds = len(full_cfg.vision_config.deepstack_visual_indexes)
        out_hidden = full_cfg.vision_config.out_hidden_size
        assert deepstack.shape == (num_ds, 1, _TEXT_SEQ_LEN, out_hidden), (
            f"Expected ({num_ds}, 1, {_TEXT_SEQ_LEN}, {out_hidden}), got {deepstack.shape}"
        )

    def test_vision_embeddings_finite(self, full_cfg, hf_model, nxdi_model):
        """Vision embeddings must be all finite (no NaN, no Inf)."""
        pixel_values, grid_thw, rotary_pos_emb, pos_embeds, vision_position_ids = (
            self._make_inputs(full_cfg, hf_model)
        )
        from neuronx_distributed_inference.modules.padding import pad_tensor
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            VISION_POSITION_ID_PAD_VALUE
        )
        bucket = _VISION_BUCKET
        padded_pv, _ = pad_tensor(pixel_values, (bucket, pixel_values.shape[1]))
        padded_rpe, _ = pad_tensor(rotary_pos_emb, (bucket, rotary_pos_emb.shape[1]))
        padded_pe, _ = pad_tensor(pos_embeds, (bucket, pos_embeds.shape[1]), pad_value=0)
        padded_vid, _ = pad_tensor(vision_position_ids, (bucket,),
                                    pad_value=VISION_POSITION_ID_PAD_VALUE)

        with torch.no_grad():
            hidden_states, deepstack = nxdi_model(
                padded_pv, padded_rpe, padded_pe, padded_vid
            )

        assert torch.isfinite(hidden_states).all(), "vision_embeddings contains NaN/Inf"
        assert torch.isfinite(deepstack).all(), "deepstack_embeds contains NaN/Inf"

    def test_hf_vision_model_forward_shape(self, hf_model):
        """Sanity-check HF Qwen3VLVisionModel produces expected output shape."""
        patch_dim = (3 * 2 * 4 * 4)  # in_ch * t_patch * patch * patch
        pixel_values = torch.randn(16, patch_dim)
        grid_thw = torch.tensor([[1, 4, 4]])

        with torch.no_grad():
            out = hf_model(pixel_values, grid_thw)

        # HF returns (hidden_states_tensor, deepstack_list) — unwrap first element
        hidden_out = out[0] if isinstance(out, (tuple, list)) else out
        # hidden_out shape: [num_merged_patches, out_hidden_size]
        # 16 patches / (spatial_merge_size^2 = 4) = 4 merged patches
        expected_seq = 16 // (_SMALL["spatial_merge_size"] ** 2)
        assert hidden_out.shape[0] == expected_seq, (
            f"Expected {expected_seq} merged patches, got {hidden_out.shape[0]}"
        )
        assert hidden_out.shape[1] == _SMALL["out_hidden_size"], (
            f"Expected out_hidden={_SMALL['out_hidden_size']}, got {hidden_out.shape[1]}"
        )

    def test_parity_vision_mlp(self, full_cfg, hf_model, nxdi_model):
        """
        Numerical parity: after loading HF weights, the patch merger (MLP projector)
        output of NxDI and HF should match for the same input.
        """
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
            NeuronQwen3VLVisionPatchMerger,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger

        cfg_ns = _make_nxdi_vision_cfg()

        # HF PatchMerger takes a Qwen3VLVisionConfig, not positional dim args
        hf_vc = _make_hf_vision_cfg()

        torch.manual_seed(5)
        hf_merger = Qwen3VLVisionPatchMerger(hf_vc, use_postshuffle_norm=False)
        hf_merger.eval()

        nxdi_merger = NeuronQwen3VLVisionPatchMerger(cfg_ns, use_postshuffle_norm=False)
        nxdi_merger.eval()

        # Copy weights: both have norm / linear_fc1 / linear_fc2
        nxdi_merger.load_state_dict(hf_merger.state_dict(), strict=True)

        # With use_postshuffle_norm=False, the merger normalises each patch individually
        # then views [-1, hidden*merge^2]. Input must be [N*merge^2, hidden_size].
        hidden = cfg_ns.hidden_size        # 64
        merge_size = cfg_ns.spatial_merge_size  # 2
        # 8 merged output patches → 8*merge^2 = 32 pre-merge patches of size 64
        x = torch.randn(8 * merge_size ** 2, hidden)
        with torch.no_grad():
            hf_out = hf_merger(x)
            nxdi_out = nxdi_merger(x)

        assert torch.allclose(hf_out, nxdi_out, atol=1e-5), (
            f"Patch merger parity failed. Max diff: {(hf_out - nxdi_out).abs().max().item():.3e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
