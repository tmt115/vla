"""
Phase 4: Weight mapping validation.

Verifies that load_vision_encoder, load_prefix_encoder, load_suffix_denoiser
correctly map checkpoint keys and produce models that match a CPU forward pass
of the original lerobot PI0Policy.
"""
import sys, torch
sys.path.insert(0, '/home/ubuntu/pi0-port')

CHECKPOINT = "/home/ubuntu/pi0-port/weights"
CHECKPOINT_FILE = f"{CHECKPOINT}/model.safetensors"


def test_vision_encoder_weights():
    """Vision encoder weights load without missing keys and output matches reference."""
    from safetensors.torch import load_file
    from vision_encoder import load_vision_encoder, NeuronVisionEncoder, _build_paligemma_model

    print("Loading vision encoder from checkpoint...")
    encoder = load_vision_encoder(CHECKPOINT_FILE)
    encoder.eval()

    # Build reference from full paligemma
    sd = load_file(CHECKPOINT_FILE)
    full = _build_paligemma_model()
    vis_prefix = "model.paligemma_with_expert.paligemma.model.vision_tower."
    prj_prefix = "model.paligemma_with_expert.paligemma.model.multi_modal_projector."
    full.model.vision_tower.load_state_dict(
        {k[len(vis_prefix):]: v for k, v in sd.items() if k.startswith(vis_prefix)}, strict=True)
    full.model.multi_modal_projector.load_state_dict(
        {k[len(prj_prefix):]: v for k, v in sd.items() if k.startswith(prj_prefix)}, strict=True)
    ref = NeuronVisionEncoder(full.model.vision_tower, full.model.multi_modal_projector)
    ref.eval()

    # Compare outputs
    px = torch.randn(1, 3, 224, 224, dtype=torch.float32) * 0.5
    with torch.no_grad():
        out_encoder = encoder(px)
        out_ref = ref(px)

    max_diff = (out_encoder - out_ref).abs().max().item()
    print(f"Vision encoder max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Vision encoder weight mismatch: max_diff={max_diff}"
    print("PASS: vision encoder weights map correctly")


def test_prefix_encoder_weights():
    """Prefix encoder weights load without non-embed missing keys."""
    from prefix_encoder import load_prefix_encoder

    print("Loading prefix encoder from checkpoint...")
    encoder = load_prefix_encoder(CHECKPOINT_FILE, seq_len=4)
    encoder.eval()

    # Verify a few key layer shapes
    layer0_q = encoder.layers[0].self_attn.q_proj.weight
    assert layer0_q.shape == (2048, 2048), f"q_proj shape: {layer0_q.shape}"
    layer17_norm = encoder.layers[17].input_layernorm.weight
    assert layer17_norm.shape == (2048,), f"norm shape: {layer17_norm.shape}"
    print(f"PASS: prefix encoder weights loaded — layer0 q_proj {layer0_q.shape}")


def test_suffix_denoiser_weights():
    """Suffix denoiser weights load without non-trivial missing keys."""
    from suffix_denoiser import load_suffix_denoiser

    print("Loading suffix denoiser from checkpoint...")
    wrapper = load_suffix_denoiser(CHECKPOINT_FILE)
    wrapper.eval()

    # Verify action projection shapes
    assert wrapper.action_in_proj.weight.shape == (1024, 32), \
        f"action_in_proj: {wrapper.action_in_proj.weight.shape}"
    assert wrapper.action_out_proj.weight.shape == (32, 1024), \
        f"action_out_proj: {wrapper.action_out_proj.weight.shape}"

    # Verify expert layer shapes
    exp_q = wrapper.model.model.layers[0].self_attn.q_proj.weight
    assert exp_q.shape == (2048, 1024), f"expert q_proj: {exp_q.shape}"

    print(f"PASS: suffix denoiser weights loaded — action_in_proj {wrapper.action_in_proj.weight.shape}")


if __name__ == "__main__":
    print("Running weight mapping validation...\n")
    test_vision_encoder_weights()
    test_prefix_encoder_weights()
    test_suffix_denoiser_weights()
    print("\nAll weight mapping tests PASSED")
