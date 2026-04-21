"""
Test NeuronVisionEncoderBlock correctness against SmolVisionEncoderReference.

Validates that the Neuron vision encoder block produces the same
[B, 64, 960] embedding tensor as the PyTorch reference when given the same weights.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEROBOT_PATH = '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages'
if LEROBOT_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_PATH)

import torch

from config_constants import (
    BATCH_SIZE,
    VISION_IMAGE_SIZE,
    VISION_TOKENS_PER_CAMERA,
    VISION_CONNECTOR_OUTPUT_DIM,
)
from vision_encoder_block import NeuronVisionEncoderBlock, SmolVisionEncoderReference, get_vision_weight_mapping


def test_vision_encoder_parity():
    """
    CPU parity test: NeuronVisionEncoderBlock matches SmolVisionEncoderReference
    with the same bfloat16 random weights.

    Strategy: initialize reference (HF wrapper), copy its weights into the
    neuron block via the weight mapping, run both with the same input,
    check max absolute difference.
    """
    print("\n" + "=" * 70)
    print("VISION ENCODER CPU PARITY TEST")
    print("=" * 70)

    B = BATCH_SIZE
    H = W = VISION_IMAGE_SIZE
    atol = 1e-2  # bfloat16 has ~1/128 precision; SigLIP has 12 layers of accumulation

    # ── Build reference block with random bfloat16 weights ──────────────────
    torch.manual_seed(42)
    ref = SmolVisionEncoderReference()
    for p in ref.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)
    ref = ref.bfloat16().eval()

    # ── Build neuron block and sync weights from reference ───────────────────
    neuron = NeuronVisionEncoderBlock()
    wm = get_vision_weight_mapping()  # {ref_key: neuron_key}

    ref_sd = ref.state_dict()
    neu_sd = neuron.state_dict()

    synced = 0
    for ref_key, neu_key in wm.items():
        if ref_key not in ref_sd:
            print(f"  MISS ref: {ref_key}")
            continue
        if neu_key not in neu_sd:
            print(f"  MISS neuron: {neu_key}")
            continue
        neu_sd[neu_key] = ref_sd[ref_key].clone()
        synced += 1
    neuron.load_state_dict(neu_sd)
    neuron = neuron.bfloat16().eval()
    print(f"  Synced {synced}/{len(wm)} weight tensors")

    # ── Run both with same input ─────────────────────────────────────────────
    torch.manual_seed(7)
    pixel = torch.randn(B, 3, H, W, dtype=torch.bfloat16)

    with torch.no_grad():
        out_ref   = ref(pixel)
        out_neuron = neuron(pixel)

    diff = (out_ref - out_neuron).abs().max().item()
    mean_ref = out_ref.float().mean().item()

    print(f"  Reference output:    shape={out_ref.shape}, mean={mean_ref:.4f}")
    print(f"  Neuron output:       shape={out_neuron.shape}")
    print(f"  Max absolute diff:   {diff:.2e}  (atol={atol})")

    assert out_ref.shape == (B, VISION_TOKENS_PER_CAMERA, VISION_CONNECTOR_OUTPUT_DIM), \
        f"Wrong ref shape: {out_ref.shape}"
    assert out_neuron.shape == out_ref.shape, \
        f"Shape mismatch: {out_neuron.shape} vs {out_ref.shape}"
    assert diff < atol, f"Max diff {diff:.2e} exceeds atol {atol}"

    print(f"\n✓ PASSED  (max_diff={diff:.2e} < atol={atol})")
    return diff


def test_vision_encoder_shapes():
    """Quick shape-only smoke test (no weight sync, no reference model)."""
    print("\nVISION ENCODER SHAPE SMOKE TEST")
    neuron = NeuronVisionEncoderBlock().eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = neuron(x)
    assert out.shape == (1, 64, 960), f"Expected (1,64,960), got {out.shape}"
    print(f"  Shape OK: {out.shape}")
    print("✓ PASSED")


if __name__ == "__main__":
    test_vision_encoder_shapes()
    diff = test_vision_encoder_parity()
    print(f"\nAll vision encoder tests passed. Max weight-synced diff: {diff:.2e}")
