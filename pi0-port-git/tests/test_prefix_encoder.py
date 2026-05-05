"""CPU tests for NeuronPrefixEncoder."""
import sys, torch
sys.path.insert(0, '/home/ubuntu/pi0-port')


def test_prefix_encoder_tiny():
    """Test with tiny Gemma config for speed."""
    from prefix_encoder import NeuronPrefixEncoder

    from transformers.models.gemma.modeling_gemma import GemmaConfig
    from lerobot.policies.pi_gemma import PiGemmaModel

    cfg = GemmaConfig(
        head_dim=16, hidden_size=32, intermediate_size=64,
        num_attention_heads=2, num_hidden_layers=2, num_key_value_heads=1,
        vocab_size=256, hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6, use_adarms=False, adarms_cond_dim=None,
    )
    cfg._attn_implementation = "eager"

    SEQ = 8
    gemma = PiGemmaModel(cfg)
    encoder = NeuronPrefixEncoder(gemma, seq_len=SEQ)
    encoder.eval()

    prefix_embs = torch.randn(1, SEQ, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        kv = encoder(prefix_embs)

    expected = (2, 2, 1, SEQ, 1, 16)   # [layers, 2, B, seq, kv_heads, head_dim]
    print(f"Tiny output shape: {tuple(kv.shape)}, expected: {expected}")
    assert tuple(kv.shape) == expected, f"Shape mismatch: {tuple(kv.shape)}"
    assert not torch.isnan(kv).any(), "NaN in output"
    print(f"  dtype: {kv.dtype} (bfloat16 when loaded from checkpoint, float32 with random init)")
    print("PASS: tiny config test")


def test_prefix_encoder_full_dims():
    """Test with real config dims but short sequence for speed."""
    from prefix_encoder import NeuronPrefixEncoder, build_gemma_model

    SEQ = 4
    gemma = build_gemma_model()
    encoder = NeuronPrefixEncoder(gemma, seq_len=SEQ)
    encoder.eval()

    prefix_embs = torch.randn(1, SEQ, 2048, dtype=torch.bfloat16)
    with torch.no_grad():
        kv = encoder(prefix_embs)

    expected = (18, 2, 1, SEQ, 1, 256)
    print(f"Full-dim output shape: {tuple(kv.shape)}, expected: {expected}")
    assert tuple(kv.shape) == expected, f"Shape mismatch"
    assert not torch.isnan(kv).any(), "NaN in output"
    print("PASS: full-dim shape test")


if __name__ == "__main__":
    print("Running prefix encoder CPU tests...")
    test_prefix_encoder_tiny()
    test_prefix_encoder_full_dims()
    print("\nAll prefix encoder tests PASSED")
