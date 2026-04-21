"""
Test NeuronVLMBackboneBlock correctness against SmolVLMBackboneReference.

Validates that the Neuron VLM backbone block produces the same [B, 113, 5120]
conditioning tensor as the PyTorch reference when given the same weights.
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
    PREFIX_SEQ_LEN,
    HIDDEN_SIZE,
    NUM_HIDDEN_LAYERS,
    CONDITIONING_HIDDEN_SIZE,
)
from vlm_backbone_block import NeuronVLMBackboneBlock, SmolVLMBackboneReference


def build_weight_mapping():
    """Build ref_key → neuron_key mapping."""
    mapping = {}
    for i in range(NUM_HIDDEN_LAYERS):
        ref_p = f"model.vlm.model.text_model.layers.{i}"
        neu_p = f"layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mapping[f"{ref_p}.self_attn.{proj}.weight"] = f"{neu_p}.self_attn.{proj}.weight"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            mapping[f"{ref_p}.mlp.{proj}.weight"] = f"{neu_p}.mlp.{proj}.weight"
        mapping[f"{ref_p}.input_layernorm.weight"]          = f"{neu_p}.input_layernorm.weight"
        mapping[f"{ref_p}.post_attention_layernorm.weight"] = f"{neu_p}.post_attention_layernorm.weight"
    mapping["model.vlm.model.text_model.norm.weight"] = "norm.weight"
    return mapping


WEIGHT_MAPPING = build_weight_mapping()


def test_vlm_backbone_parity():
    """
    CPU parity test: NeuronVLMBackboneBlock matches SmolVLMBackboneReference
    with the same bfloat16 random weights.
    """
    print("\n" + "=" * 70)
    print("VLM BACKBONE CPU PARITY TEST")
    print("=" * 70)

    B = BATCH_SIZE
    L = PREFIX_SEQ_LEN
    H = HIDDEN_SIZE
    atol = 5e-2  # 16-layer transformer with GQA; bfloat16 accumulation tolerance

    # ── Build reference with random bfloat16 weights ─────────────────────────
    torch.manual_seed(42)
    ref = SmolVLMBackboneReference()
    for p in ref.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)
    ref = ref.bfloat16().eval()

    # ── Build neuron block and sync weights ──────────────────────────────────
    neuron = NeuronVLMBackboneBlock()
    wm = WEIGHT_MAPPING

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
    prefix    = torch.randn(B, L, H, dtype=torch.bfloat16)
    attn_mask = torch.ones(B, L, L, dtype=torch.bool)
    pos_ids   = torch.arange(L).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        out_ref    = ref(prefix)
        out_neuron = neuron(prefix, attn_mask, pos_ids)

    diff = (out_ref - out_neuron).abs().max().item()
    print(f"  Reference output:    shape={out_ref.shape}, mean={out_ref.float().mean().item():.4f}")
    print(f"  Neuron output:       shape={out_neuron.shape}")
    print(f"  Max absolute diff:   {diff:.2e}  (atol={atol})")

    assert out_ref.shape    == (B, L, CONDITIONING_HIDDEN_SIZE), f"Bad ref shape: {out_ref.shape}"
    assert out_neuron.shape == out_ref.shape, "Shape mismatch"
    assert diff < atol, f"Max diff {diff:.2e} exceeds atol {atol}"

    print(f"\n✓ PASSED  (max_diff={diff:.2e} < atol={atol})")
    return diff


def test_vlm_backbone_shapes():
    """Quick shape smoke test."""
    print("\nVLM BACKBONE SHAPE SMOKE TEST")
    B, L, H = 1, PREFIX_SEQ_LEN, HIDDEN_SIZE
    neuron    = NeuronVLMBackboneBlock().eval()
    prefix    = torch.randn(B, L, H)
    attn_mask = torch.ones(B, L, L, dtype=torch.bool)
    pos_ids   = torch.arange(L).unsqueeze(0)
    with torch.no_grad():
        out = neuron(prefix, attn_mask, pos_ids)
    assert out.shape == (B, L, CONDITIONING_HIDDEN_SIZE), f"Expected ({B},{L},5120), got {out.shape}"
    print(f"  Shape OK: {out.shape}")
    print("✓ PASSED")


if __name__ == "__main__":
    test_vlm_backbone_shapes()
    diff = test_vlm_backbone_parity()
    print(f"\nAll VLM backbone tests passed. Max weight-synced diff: {diff:.2e}")
