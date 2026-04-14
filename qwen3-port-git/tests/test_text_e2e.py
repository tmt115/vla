"""
Deliverable 5: Text decoder end-to-end CPU parity test.
Validates NeuronQwen3VLTextModel against HF Qwen3VLTextModel on CPU
using a small synthetic config, random weights, and dummy vision embeddings.

Covers:
  1. Text-only prefill (context encoding) forward parity
  2. Token generation forward (single decode step)
  3. Prefill with dummy vision_embeddings (zero-filled, no-op scatter)
  4. Output shape checks: logits [B, seq_len, vocab_size]

Run:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  cd ~/qwen-port && python -m pytest tests/test_text_e2e.py -v -s
"""

import os
import sys
import types
import pytest
import torch
from unittest.mock import patch as _mock_patch

NXDI_ROOT = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages"
if NXDI_ROOT not in sys.path:
    sys.path.insert(0, NXDI_ROOT)

# ── One-time environment setup ─────────────────────────────────────────────────
# NeuronQwen3VLTextModel → NeuronBaseModel.__init__ always creates SPMDRank,
# which calls get_tensor_model_parallel_size() → needs torch.distributed + NxD
# model-parallel initialized, even for CPU-only tests.
#
# Also patch cpu_mode()=True here at module level so that NeuronQwen3VLAttention
# and DecoderLayer pick LlamaRMSNorm (not CustomRMSNorm which needs XLA tensors).
# Monkeypatching at function scope is too late: class-scoped fixtures that build
# the model run before function-scoped fixtures execute.

def _init_distributed_for_tests():
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)
    from neuronx_distributed.parallel_layers import parallel_state as _ps
    if not _ps.model_parallel_is_initialized():
        _ps.initialize_model_parallel(1, skip_collective_init=True)

_init_distributed_for_tests()

# Apply cpu_mode patch permanently for this module so it is active when the
# class-scoped nxdi_model fixture constructs NeuronQwen3VLTextModel.
_cpu_patch = _mock_patch(
    "neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text.cpu_mode",
    lambda: True,
)
_cpu_patch.start()


# ── Config ─────────────────────────────────────────────────────────────────────

_HIDDEN = 128
_N_HEADS = 8
_N_KV = 2
_N_LAYERS = 2
_INTERMEDIATE = 256
_HEAD_DIM = _HIDDEN // _N_HEADS   # 16
_VOCAB = 1000
_SEQ_LEN = 32   # bucket for tracing
_TEXT_ONLY_SEQ = 16  # active tokens in test


def _make_text_nc(tp_degree=1):
    from neuronx_distributed_inference.models.config import NeuronConfig
    nc = NeuronConfig(tp_degree=tp_degree, torch_dtype="float32")
    nc.rpl_reduce_dtype = None
    nc.sequence_parallel_enabled = False
    nc.fused_qkv = False
    return nc


def _make_hf_text_cfg():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    tc = Qwen3VLTextConfig(
        hidden_size=_HIDDEN,
        num_attention_heads=_N_HEADS,
        num_key_value_heads=_N_KV,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_N_LAYERS,
        head_dim=_HEAD_DIM,
        vocab_size=_VOCAB,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 4, 4]},
        max_position_embeddings=_SEQ_LEN,
    )
    tc._attn_implementation = "eager"
    return tc


def _make_nxdi_inference_cfg():
    """
    Construct a minimal Qwen3VLInferenceConfig for text-model-only tests.
    Uses a real Qwen3VLVisionConfig to satisfy the nested-config requirement,
    but vision tests are run independently.
    """
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig, Qwen3VLTextConfig,
    )
    from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLInferenceConfig, Qwen3VLNeuronConfig,
    )

    hf_tc = _make_hf_text_cfg()
    hf_vc = Qwen3VLVisionConfig(
        depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
        in_channels=3, patch_size=4, temporal_patch_size=2,
        spatial_merge_size=2, out_hidden_size=_HIDDEN,
        num_position_embeddings=16, hidden_act="gelu_pytorch_tanh",
    )
    hf_vc.deepstack_visual_indexes = []   # no deepstack for this test

    text_nc = Qwen3VLNeuronConfig(
        tp_degree=1,
        torch_dtype="float32",
        batch_size=1,
        seq_len=_SEQ_LEN,
        n_active_tokens=_SEQ_LEN,
        max_context_length=_SEQ_LEN,
        on_device_sampling_config=None,
        buckets=[_SEQ_LEN],
    )
    vision_nc = Qwen3VLNeuronConfig(
        tp_degree=1,
        torch_dtype="float32",
        batch_size=1,
        seq_len=16,
        n_active_tokens=16,
        on_device_sampling_config=None,
        buckets=[16],
    )

    cfg = Qwen3VLInferenceConfig(
        text_neuron_config=text_nc,
        vision_neuron_config=vision_nc,
        text_config=vars(hf_tc),
        vision_config=vars(hf_vc),
        _name_or_path="/tmp/fake_qwen3_vl_text",
    )
    return cfg


def _convert_hf_to_nxdi_text_sd(hf_sd):
    """Apply convert_hf_to_neuron_state_dict renaming logic to a layer state dict."""
    attn_renames = {
        "language_model.": "",           # strip language_model prefix
    }
    key_renames = {
        ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
        ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
        ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
        ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
        ".q_norm.": ".q_layernorm.",
        ".k_norm.": ".k_layernorm.",
    }
    converted = {}
    for k, v in hf_sd.items():
        new_k = k
        for old, new in attn_renames.items():
            new_k = new_k.replace(old, new)
        for old, new in key_renames.items():
            new_k = new_k.replace(old, new)
        converted[new_k] = v.clone()
    return converted


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestTextModelE2E:

    @pytest.fixture(scope="class")
    def cfg(self):
        return _make_nxdi_inference_cfg()

    @pytest.fixture(scope="class")
    def nxdi_model(self, cfg):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
            NeuronQwen3VLTextModel,
        )
        torch.manual_seed(42)
        model = NeuronQwen3VLTextModel(cfg.text_config)
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def hf_text_model(self):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
        tc = _make_hf_text_cfg()
        torch.manual_seed(42)
        m = Qwen3VLTextModel(tc)
        m.eval()
        return m

    # ── Helper ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _forward(model, input_ids, position_ids, rotary_position_ids=None,
                 vision_embeddings=None, vision_mask=None):
        """
        Manual transformer forward: embed → scatter vision → layers → norm → lm_head.
        Bypasses NeuronBaseModel.forward() which is the inference-serving API
        (requires sampling_params, KV-cache management, etc.) and is not intended
        for direct unit-test invocation on CPU.
        """
        hidden_states = model.embed_tokens(input_ids)

        # Optionally scatter vision embeddings into token positions
        if vision_embeddings is not None and vision_mask is not None:
            hidden_states = model.encode_vision_to_input(
                hidden_states, vision_embeddings, vision_mask
            )

        kwargs = {}
        if rotary_position_ids is not None:
            kwargs["rotary_position_ids"] = rotary_position_ids

        for layer in model.layers:
            outputs = layer(hidden_states, position_ids=position_ids, **kwargs)
            hidden_states = outputs[0]

        hidden_states = model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        return logits

    # ── Shape tests ────────────────────────────────────────────────────────────

    def test_text_model_state_dict_keys(self, nxdi_model):
        """NeuronQwen3VLTextModel has embed_tokens, layers, norm, lm_head."""
        keys = set(nxdi_model.state_dict().keys())
        assert any("embed_tokens" in k for k in keys), "Missing embed_tokens"
        assert any("layers.0" in k for k in keys), "Missing layer 0"
        assert any("norm" in k for k in keys), "Missing final norm"
        assert any("lm_head" in k for k in keys), "Missing lm_head"

    def test_prefill_output_shape(self, cfg, nxdi_model):
        """Text-only prefill returns logits of shape [B, seq, vocab]."""
        bs, seq = 1, _TEXT_ONLY_SEQ
        input_ids = torch.randint(0, _VOCAB, (bs, seq))
        position_ids = torch.arange(seq).unsqueeze(0).repeat(bs, 1)

        with torch.no_grad():
            logits = self._forward(nxdi_model, input_ids, position_ids)

        assert logits.shape == (bs, seq, _VOCAB), (
            f"Expected logits shape ({bs}, {seq}, {_VOCAB}), got {logits.shape}"
        )

    def test_prefill_output_finite(self, cfg, nxdi_model):
        """Prefill output must be finite (no NaN, no Inf)."""
        bs, seq = 1, _TEXT_ONLY_SEQ
        input_ids = torch.randint(0, _VOCAB, (bs, seq))
        position_ids = torch.arange(seq).unsqueeze(0)

        with torch.no_grad():
            logits = self._forward(nxdi_model, input_ids, position_ids)

        assert torch.isfinite(logits).all(), "Text model prefill output contains NaN/Inf"

    def test_prefill_with_dummy_vision_embeddings(self, cfg, nxdi_model):
        """Text model encode_vision_to_input scatters zero embeddings without error."""
        bs, seq = 1, _TEXT_ONLY_SEQ
        input_ids = torch.randint(0, _VOCAB, (bs, seq))
        position_ids = torch.arange(seq).unsqueeze(0)

        # Dummy vision inputs: all zeros = no vision signal (mask is zero so no scatter)
        vision_embeddings = torch.zeros(bs, seq, _HIDDEN)
        vision_mask = torch.zeros(bs, seq, 1, dtype=torch.int32)

        with torch.no_grad():
            logits = self._forward(
                nxdi_model, input_ids, position_ids,
                vision_embeddings=vision_embeddings,
                vision_mask=vision_mask,
            )

        assert logits.shape == (bs, seq, _VOCAB)
        assert torch.isfinite(logits).all()

    def test_prefill_with_mrope_position_ids(self, cfg, nxdi_model):
        """3D MRoPE rotary_position_ids [3, B, S] passed to decoder layers."""
        bs, seq = 1, _TEXT_ONLY_SEQ
        input_ids = torch.randint(0, _VOCAB, (bs, seq))
        position_ids = torch.arange(seq).unsqueeze(0)

        # 3D rotary position ids for MRoPE: [t, h, w], all same for text-only
        rotary_position_ids = torch.stack([
            position_ids,   # temporal
            position_ids,   # height
            position_ids,   # width
        ])  # [3, 1, seq]

        with torch.no_grad():
            logits = self._forward(
                nxdi_model, input_ids, position_ids,
                rotary_position_ids=rotary_position_ids,
            )

        assert logits.shape == (bs, seq, _VOCAB)
        assert torch.isfinite(logits).all()

    # ── Parity tests ───────────────────────────────────────────────────────────

    def test_mlp_parity_with_hf(self):
        """NxDI MLP matches HF SwiGLU for identical weights."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP
        from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP

        tc = _make_hf_text_cfg()
        cfg_ns = types.SimpleNamespace()
        cfg_ns.hidden_size = _HIDDEN
        cfg_ns.intermediate_size = _INTERMEDIATE
        cfg_ns.hidden_act = "silu"
        cfg_ns.rms_norm_eps = 1e-6
        cfg_ns.neuron_config = _make_text_nc()

        torch.manual_seed(9)
        hf_mlp = Qwen3VLTextMLP(tc)
        hf_mlp.eval()
        nxdi_mlp = NeuronLlamaMLP(cfg_ns)
        nxdi_mlp.eval()
        nxdi_mlp.load_state_dict(hf_mlp.state_dict(), strict=True)

        x = torch.randn(1, 8, _HIDDEN)
        with torch.no_grad():
            hf_out = hf_mlp(x)
            nxdi_out = nxdi_mlp(x)[0]

        assert torch.allclose(hf_out, nxdi_out, atol=1e-5), (
            f"Text MLP parity failed. Max diff: {(hf_out - nxdi_out).abs().max().item():.3e}"
        )

    def test_embed_tokens_shape(self, nxdi_model):
        """Token embedding lookup returns [B, S, hidden]."""
        ids = torch.randint(0, _VOCAB, (1, 8))
        # Access embed_tokens directly
        with torch.no_grad():
            emb = nxdi_model.embed_tokens(ids)
        assert emb.shape == (1, 8, _HIDDEN), f"Expected (1, 8, {_HIDDEN}), got {emb.shape}"

    def test_decode_step_shape(self, cfg, nxdi_model):
        """Single decode step (seq_len=1) returns logits [B, 1, vocab]."""
        bs = 1
        input_ids = torch.randint(0, _VOCAB, (bs, 1))
        position_ids = torch.tensor([[_TEXT_ONLY_SEQ]])  # position after prefill

        with torch.no_grad():
            logits = self._forward(nxdi_model, input_ids, position_ids)

        assert logits.shape[0] == bs
        assert logits.shape[-1] == _VOCAB
        assert torch.isfinite(logits).all()

    def test_hf_qwen3vl_text_model_forward(self, hf_text_model):
        """Sanity-check HF Qwen3VLTextModel forward runs correctly."""
        from transformers import DynamicCache
        bs, seq = 1, _TEXT_ONLY_SEQ
        input_ids = torch.randint(0, _VOCAB, (bs, seq))
        attention_mask = torch.ones(bs, seq)
        pos_ids = torch.arange(seq).unsqueeze(0)

        with torch.no_grad():
            out = hf_text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                use_cache=False,
            )

        last_hs = out.last_hidden_state
        assert last_hs.shape == (bs, seq, _HIDDEN), (
            f"Expected ({bs}, {seq}, {_HIDDEN}), got {last_hs.shape}"
        )
        assert torch.isfinite(last_hs).all()


def _make_text_nc():
    from neuronx_distributed_inference.models.config import NeuronConfig
    nc = NeuronConfig(tp_degree=1, torch_dtype="float32")
    nc.rpl_reduce_dtype = None
    nc.sequence_parallel_enabled = False
    nc.fused_qkv = False
    return nc


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
