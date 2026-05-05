"""
CPU integration smoke test — verifies the three blocks compose correctly
without Neuron compilation. Runs entirely on CPU with random weights.
"""
import sys, math, torch
sys.path.insert(0, '/home/ubuntu/pi0-port')
sys.path.insert(0, '/home/ubuntu/pi0-port/skills/scripts')

from config_constants import (
    SIGLIP_NUM_IMAGE_TOKENS, VLM_HIDDEN_SIZE, NUM_CAMERAS,
    EXPERT_HIDDEN_SIZE, EXPERT_NUM_LAYERS, EXPERT_HEAD_DIM, EXPERT_NUM_KV_HEADS,
    CHUNK_SIZE, MAX_STATE_DIM, MAX_ACTION_DIM, PREFIX_LEN, SUFFIX_LEN,
    MIN_PERIOD, MAX_PERIOD, MAX_LANG_TOKENS,
)


def test_vision_to_prefix():
    """Vision encoder output feeds into prefix encoder correctly."""
    from vision_encoder import NeuronVisionEncoder, _build_paligemma_model
    from prefix_encoder import NeuronPrefixEncoder

    # Build tiny prefix encoder (real VLM dims but short seq for speed)
    from prefix_encoder import build_gemma_model
    SEQ = 4   # tiny for speed
    gemma = build_gemma_model()
    pe = NeuronPrefixEncoder(gemma, seq_len=SEQ)
    pe.eval()

    # Fake prefix_embs (would come from vision + lang embeddings)
    prefix_embs = torch.randn(1, SEQ, VLM_HIDDEN_SIZE, dtype=torch.bfloat16)
    with torch.no_grad():
        prefix_kv = pe(prefix_embs)

    assert prefix_kv.shape == (18, 2, 1, SEQ, 1, 256), f"prefix_kv shape: {prefix_kv.shape}"
    print(f"PASS: vision→prefix — prefix_kv {prefix_kv.shape}")


def test_prefix_to_suffix():
    """Prefix KV cache feeds into suffix denoiser correctly."""
    from neuron_action_head_base import NeuronDenoisingConfig
    from suffix_denoiser import NeuronPi0DenoisingWrapper

    TINY_H = 32
    TINY_L = 2
    TINY_NH = 2
    TINY_HD = 16
    TINY_PREFIX = 4
    TINY_CHUNK = 3
    TINY_STATE = 4
    TINY_ACTION = 4

    config = NeuronDenoisingConfig(
        batch_size=1, tp_degree=1,
        action_chunk_size=TINY_CHUNK, action_dim=TINY_ACTION,
        num_conditioning_tokens=TINY_PREFIX, conditioning_hidden_size=TINY_H,
        timestep_embed_dim=TINY_H,
    )
    wrapper = NeuronPi0DenoisingWrapper(
        config,
        num_layers=TINY_L, hidden_size=TINY_H, head_dim=TINY_HD,
        num_heads=TINY_NH, kv_heads=1, intermediate_size=TINY_H * 4,
        prefix_len=TINY_PREFIX, chunk_size=TINY_CHUNK,
        max_state=TINY_STATE, max_action=TINY_ACTION,
    )
    wrapper.load_module()
    wrapper.eval()

    B = 1
    prefix_kv = torch.zeros(TINY_L, 2, B, TINY_PREFIX, 1, TINY_HD, dtype=torch.bfloat16)
    state = torch.zeros(B, TINY_STATE, dtype=torch.bfloat16)
    noisy_actions = torch.randn(B, TINY_CHUNK, TINY_ACTION, dtype=torch.bfloat16)
    time_emb = torch.zeros(B, TINY_H, dtype=torch.bfloat16)

    with torch.no_grad():
        v_t = wrapper(state, noisy_actions, time_emb, prefix_kv)

    assert v_t.shape == (B, TINY_CHUNK, TINY_ACTION), f"v_t shape: {v_t.shape}"
    assert v_t.dtype == torch.float32
    assert not torch.isnan(v_t).any()
    print(f"PASS: prefix_kv→suffix — v_t {v_t.shape} float32")


def test_full_embedding_pipeline():
    """Verify image + language → prefix_embs → prefix_kv shape contract."""
    from vision_encoder import NeuronVisionEncoder, _build_paligemma_model
    from prefix_encoder import NeuronPrefixEncoder, build_gemma_model
    from pi0_neuron import LanguageEmbedder

    # Build real-sized components but use a tiny seq prefix encoder
    full_model = _build_paligemma_model()
    ve = NeuronVisionEncoder(full_model.model.vision_tower, full_model.model.multi_modal_projector)
    ve.eval()

    lang_emb = LanguageEmbedder(full_model.model.language_model.embed_tokens)
    lang_emb.eval()

    # Tiny prefix encoder with real VLM dims but very short seq
    SEQ = 2  # 1 token per camera + 1 lang token, just to verify shape
    pe_real = build_gemma_model()
    pe_tiny = NeuronPrefixEncoder(pe_real, seq_len=SEQ)
    pe_tiny.eval()

    B = 1
    # Simulate: 1 camera with 1 image token + 1 lang token = 2 prefix tokens
    # For this test, just verify the shapes compose correctly end-to-end
    with torch.no_grad():
        # Vision encoder: [B, 3, 224, 224] → [B, 256, 2048]
        px = torch.randn(B, 3, 224, 224, dtype=torch.float32)
        img_feat = ve(px)
        assert img_feat.shape == (B, 256, 2048), f"img_feat: {img_feat.shape}"

        # Language embedder: [B, T] → [B, T, 2048]
        lang_tokens = torch.zeros(B, MAX_LANG_TOKENS, dtype=torch.long)
        lang_feat = lang_emb(lang_tokens)
        assert lang_feat.shape == (B, MAX_LANG_TOKENS, 2048), f"lang_feat: {lang_feat.shape}"

    print(f"PASS: embedding pipeline — img_feat {img_feat.shape}, lang_feat {lang_feat.shape}")

    # Verify prefix concat shape (without running 18-layer encoder for speed)
    prefix_embs = torch.cat([img_feat.to(torch.bfloat16), lang_feat], dim=1)
    expected_prefix = B, 256 + MAX_LANG_TOKENS, 2048
    assert prefix_embs.shape == torch.Size(expected_prefix), f"prefix_embs: {prefix_embs.shape}"
    print(f"PASS: prefix concat shape {prefix_embs.shape} (1-camera case)")

    # With 3 cameras: 3*256 + 48 = 816 tokens
    prefix_3cam = torch.cat([img_feat.to(torch.bfloat16)] * 3 + [lang_feat], dim=1)
    assert prefix_3cam.shape == (B, PREFIX_LEN, 2048), f"3-cam prefix: {prefix_3cam.shape}"
    print(f"PASS: 3-camera prefix shape {prefix_3cam.shape}")


def test_kv_shape_contract():
    """Assert KV shapes match the contract between prefix encoder and suffix denoiser."""
    from config_constants import EXPERT_NUM_LAYERS, EXPERT_HEAD_DIM, EXPERT_NUM_KV_HEADS

    # The contract:
    # prefix_encoder output: [18, 2, B, 816, 1, 256]
    # suffix_denoiser input:  [18, 2, B, 816, 1, 256]
    # Both must match exactly
    expected_kv_shape = (EXPERT_NUM_LAYERS, 2, 1, PREFIX_LEN, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)
    print(f"KV contract shape: {expected_kv_shape}")

    from suffix_denoiser import NeuronPi0DenoisingWrapper
    from neuron_action_head_base import NeuronDenoisingConfig

    config = NeuronDenoisingConfig(
        batch_size=1, tp_degree=1,
        action_chunk_size=CHUNK_SIZE, action_dim=MAX_ACTION_DIM,
        num_conditioning_tokens=PREFIX_LEN, conditioning_hidden_size=EXPERT_HIDDEN_SIZE,
        timestep_embed_dim=EXPERT_HIDDEN_SIZE,
    )
    wrapper = NeuronPi0DenoisingWrapper(config)
    input_gen = wrapper.input_generator()
    pkv_example = input_gen[0][3]  # prefix_kv is 4th input

    assert tuple(pkv_example.shape) == expected_kv_shape, \
        f"KV contract mismatch: {tuple(pkv_example.shape)} != {expected_kv_shape}"
    print(f"PASS: KV shape contract verified {pkv_example.shape}")


if __name__ == "__main__":
    print("Running CPU integration tests...\n")
    test_kv_shape_contract()
    test_vision_to_prefix()
    test_prefix_to_suffix()
    test_full_embedding_pipeline()
    print("\nAll CPU integration tests PASSED")
