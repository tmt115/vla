"""CPU parity test for NeuronVisionEncoder."""
import sys, math, torch
sys.path.insert(0, '/home/ubuntu/pi0-port')


def test_vision_encoder_cpu():
    """Test NeuronVisionEncoder output shape and numerical parity against reference.

    PaliGemmaModel.get_image_features() does:
      1. image_outputs = vision_tower(pixel_values)
      2. selected = image_outputs.last_hidden_state        [B, 256, 1152]
      3. features  = multi_modal_projector(selected)       [B, 256, 2048]
      4. features  = features / sqrt(hidden_size=2048)     [B, 256, 2048]  ← scaling

    NeuronVisionEncoder.forward() performs steps 1-3 only (no sqrt scaling).
    So: encoder(px) * (1/sqrt(2048)) should equal get_image_features(px).
    """
    from vision_encoder import NeuronVisionEncoder, _build_paligemma_model

    torch.manual_seed(42)

    # Build reference model
    ref_model = _build_paligemma_model()
    ref_model.eval()

    # Build NeuronVisionEncoder with the same weights
    encoder = NeuronVisionEncoder(
        vision_tower=ref_model.model.vision_tower,
        multi_modal_projector=ref_model.model.multi_modal_projector,
    )
    encoder.eval()

    # Random pixel values in [-1, 1] range (pi0 normalization)
    px = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        # get_image_features returns [B, 256, 2048] already divided by sqrt(2048)
        ref_out = ref_model.model.get_image_features(px)
        neuron_out = encoder(px)

    print(f"Reference output type: {type(ref_out)}, shape attr check...")
    if hasattr(ref_out, 'last_hidden_state'):
        ref_features = ref_out.last_hidden_state
        print(f"  last_hidden_state shape: {ref_features.shape}")
    elif hasattr(ref_out, 'pooler_output'):
        ref_features = ref_out.pooler_output
        print(f"  pooler_output shape: {ref_features.shape}")
    else:
        ref_features = ref_out
        print(f"  tensor shape: {ref_features.shape}")

    print(f"NeuronVisionEncoder output shape: {neuron_out.shape}")

    # Shape check
    assert neuron_out.shape == (1, 256, 2048), f"Expected [1,256,2048], got {neuron_out.shape}"
    assert not torch.isnan(neuron_out).any(), "NaN in NeuronVisionEncoder output"

    # get_image_features returns a plain tensor [B, 256, 2048] (already scaled)
    # NeuronVisionEncoder returns un-scaled features; apply sqrt(2048) to match
    hidden_size = 2048
    neuron_scaled = neuron_out / math.sqrt(hidden_size)

    assert ref_features.shape == (1, 256, 2048), \
        f"Expected ref_features shape [1,256,2048], got {ref_features.shape}"
    assert not torch.isnan(ref_features).any(), "NaN in reference output"

    # Numerical parity: should be bit-exact (same weights, same ops, CPU)
    max_diff = (neuron_scaled - ref_features).abs().max().item()
    print(f"Max abs diff (after sqrt scaling): {max_diff:.2e}")
    assert max_diff < 1e-5, f"Parity check failed: max_diff={max_diff:.2e}"

    print("PASS: shape correct, no NaN, numerical parity with get_image_features VERIFIED")
    print("All vision encoder CPU tests PASSED")


if __name__ == "__main__":
    test_vision_encoder_cpu()
