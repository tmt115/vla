"""
Phase 4 weight mapping validation.

Verifies that convert_hf_to_neuron_state_dict produces a complete, shape-matching
state dict for all three subgraphs, and that loading real weights produces valid
(non-NaN, non-Inf) outputs.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/smol-port-skilled
    python tests/test_phase4_weights.py
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
from safetensors.torch import load_file

CHECKPOINT = str(ROOT / "checkpoint" / "model.safetensors")

from config_constants import (
    BATCH_SIZE, VISION_IMAGE_SIZE, HIDDEN_SIZE, PREFIX_SEQ_LEN,
    NUM_TEXT_TOKENS, NUM_STATE_TOKENS, VISION_TOKENS_PER_CAMERA,
    ACTION_CHUNK_SIZE, ACTION_DIM, NUM_CONDITIONING_TOKENS, NUM_DENOISING_STEPS,
)
from modeling_smolvla_neuron import (
    NeuronSmolVLAVisionModel,
    NeuronSmolVLAVisionHead,
    NeuronSmolVLABackboneModel,
    NeuronSmolVLABackboneHead,
    NeuronSmolVLAPolicy,
)
from action_expert_block import (
    NeuronSmolVLADenoisingWrapper,
    get_hf_to_neuron_weight_mapping,
)


def load_hf_checkpoint():
    print(f"  Loading HF checkpoint from {CHECKPOINT} ...")
    sd = load_file(CHECKPOINT)
    print(f"  {len(sd)} tensors loaded.")
    return sd


def test_vision_weight_mapping():
    print("\n--- Phase 4: Vision encoder weight mapping ---")
    hf_sd = load_hf_checkpoint()

    model = NeuronSmolVLAVisionModel()
    neuron_keys = set(model.state_dict().keys())

    converted = NeuronSmolVLAVisionHead.convert_hf_to_neuron_state_dict(hf_sd, config=None)
    converted_keys = set(converted.keys())

    missing = neuron_keys - converted_keys
    extra   = converted_keys - neuron_keys

    if missing:
        print(f"  MISSING keys ({len(missing)}): {sorted(missing)[:10]}")
    if extra:
        print(f"  EXTRA   keys ({len(extra)}):   {sorted(extra)[:5]}")

    assert not missing, f"Missing {len(missing)} keys in vision conversion"
    print(f"  Key coverage: {len(neuron_keys)}/{len(neuron_keys)} ✓")

    # Shape check
    for key in neuron_keys:
        expected = model.state_dict()[key].shape
        actual   = converted[key].shape
        assert expected == actual, f"{key}: expected {expected}, got {actual}"
    print(f"  Shape check: all {len(neuron_keys)} keys match ✓")

    # Load weights and run forward
    model.load_state_dict(converted, strict=True)
    model.eval()
    pixel = torch.randn(BATCH_SIZE, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE,
                        dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(pixel)
    assert out.shape == (BATCH_SIZE, VISION_TOKENS_PER_CAMERA, HIDDEN_SIZE)
    assert not torch.isnan(out).any(), "NaN in vision encoder output"
    assert not torch.isinf(out).any(), "Inf in vision encoder output"
    print(f"  Forward pass with real weights: {tuple(out.shape)}, "
          f"mean={out.float().mean().item():.4f} ✓")
    print("  PASS")


def test_backbone_weight_mapping():
    print("\n--- Phase 4: VLM backbone weight mapping ---")
    hf_sd = load_hf_checkpoint()

    model = NeuronSmolVLABackboneModel()
    neuron_keys = set(model.state_dict().keys())

    converted = NeuronSmolVLABackboneHead.convert_hf_to_neuron_state_dict(hf_sd, config=None)
    converted_keys = set(converted.keys())

    missing = neuron_keys - converted_keys
    extra   = converted_keys - neuron_keys

    if missing:
        print(f"  MISSING keys ({len(missing)}): {sorted(missing)[:10]}")

    assert not missing, f"Missing {len(missing)} keys in backbone conversion"
    print(f"  Key coverage: {len(neuron_keys)}/{len(neuron_keys)} ✓")

    for key in neuron_keys:
        expected = model.state_dict()[key].shape
        actual   = converted[key].shape
        assert expected == actual, f"{key}: expected {expected}, got {actual}"
    print(f"  Shape check: all {len(neuron_keys)} keys match ✓")

    model.load_state_dict(converted, strict=True)
    model.eval()
    prefix    = torch.randn(BATCH_SIZE, PREFIX_SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)
    attn_mask = torch.ones(BATCH_SIZE, PREFIX_SEQ_LEN, PREFIX_SEQ_LEN, dtype=torch.bool)
    pos_ids   = torch.arange(PREFIX_SEQ_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)
    with torch.no_grad():
        out = model(prefix, attn_mask, pos_ids)
    assert out.shape == (BATCH_SIZE, NUM_CONDITIONING_TOKENS, 5120)
    assert not torch.isnan(out).any(), "NaN in backbone output"
    assert not torch.isinf(out).any(), "Inf in backbone output"
    print(f"  Forward pass with real weights: {tuple(out.shape)}, "
          f"mean={out.float().mean().item():.4f} ✓")
    print("  PASS")


def test_expert_weight_mapping():
    print("\n--- Phase 4: Action expert weight mapping ---")
    hf_sd = load_hf_checkpoint()

    model = NeuronSmolVLADenoisingWrapper()
    neuron_keys = set(model.state_dict().keys())

    hf_mapping = get_hf_to_neuron_weight_mapping()  # {hf_key: neuron_key}
    converted = {neuron_key: hf_sd[hf_key]
                 for hf_key, neuron_key in hf_mapping.items()
                 if hf_key in hf_sd}

    missing = neuron_keys - set(converted.keys())
    if missing:
        print(f"  MISSING keys ({len(missing)}): {sorted(missing)[:10]}")

    assert not missing, f"Missing {len(missing)} keys in expert conversion"
    print(f"  Key coverage: {len(neuron_keys)}/{len(neuron_keys)} ✓")

    for key in neuron_keys:
        expected = model.state_dict()[key].shape
        actual   = converted[key].shape
        assert expected == actual, f"{key}: expected {expected}, got {actual}"
    print(f"  Shape check: all {len(neuron_keys)} keys match ✓")

    model.load_state_dict(converted, strict=True)
    model.eval()

    B = BATCH_SIZE
    noisy    = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, dtype=torch.bfloat16)
    cond     = torch.randn(B, NUM_CONDITIONING_TOKENS, 5120, dtype=torch.bfloat16)
    t_emb    = torch.randn(B, 720, dtype=torch.bfloat16)
    mask     = torch.ones(B, 1, ACTION_CHUNK_SIZE, NUM_CONDITIONING_TOKENS, dtype=torch.int32)
    with torch.no_grad():
        out = model(noisy, cond, t_emb, mask)
    assert out.shape == (B, ACTION_CHUNK_SIZE, ACTION_DIM)
    assert not torch.isnan(out).any(), "NaN in expert output"
    assert not torch.isinf(out).any(), "Inf in expert output"
    print(f"  Forward pass with real weights: {tuple(out.shape)}, "
          f"mean={out.float().mean().item():.4f} ✓")
    print("  PASS")


def test_end_to_end_real_weights():
    print("\n--- Phase 4: End-to-end policy with real weights ---")
    hf_sd = load_hf_checkpoint()

    policy = NeuronSmolVLAPolicy()
    policy.compile("/tmp/test_e2e_compiled")  # builds denoising_wrapper (CPU no-op)

    # Load vision weights
    vis_sd = NeuronSmolVLAVisionHead.convert_hf_to_neuron_state_dict(hf_sd, None)
    policy.vision_model.load_state_dict(vis_sd, strict=True)

    # Load backbone weights
    bac_sd = NeuronSmolVLABackboneHead.convert_hf_to_neuron_state_dict(hf_sd, None)
    policy.backbone_model.load_state_dict(bac_sd, strict=True)

    # Load expert weights
    from action_expert_block import load_from_checkpoint
    load_from_checkpoint(policy.action_head.denoising_wrapper, CHECKPOINT)

    policy.eval()

    B = BATCH_SIZE
    pixel    = torch.randn(B, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE, dtype=torch.bfloat16)
    text_emb = torch.randn(B, NUM_TEXT_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16)
    state_emb = torch.randn(B, NUM_STATE_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16)

    actions = policy.run(pixel, text_emb, state_emb, num_denoising_steps=NUM_DENOISING_STEPS)
    assert actions.shape == (B, ACTION_CHUNK_SIZE, ACTION_DIM)
    assert not torch.isnan(actions).any(), "NaN in final actions"
    assert not torch.isinf(actions).any(), "Inf in final actions"
    print(f"  Final actions: shape={tuple(actions.shape)}, "
          f"mean={actions.float().mean().item():.4f}, "
          f"std={actions.float().std().item():.4f} ✓")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4 WEIGHT MAPPING VALIDATION")
    print("=" * 70)

    test_vision_weight_mapping()
    test_backbone_weight_mapping()
    test_expert_weight_mapping()
    test_end_to_end_real_weights()

    print("\n" + "=" * 70)
    print("ALL PHASE 4 TESTS PASSED")
    print("=" * 70)
