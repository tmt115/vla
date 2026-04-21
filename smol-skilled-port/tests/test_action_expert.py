"""
test_action_expert.py

Correctness tests for NeuronSmolVLADenoisingWrapper and NeuronSmolVLAActionHead.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/smol-port-skilled
    python tests/test_action_expert.py [--test cpu|single|loop|all]

Tests:
    cpu     — CPU interface smoke test (no hardware required)
    single  — Single denoising step correctness vs reference (atol=1e-3)
    loop    — Full 10-step denoising loop correctness (atol=1e-2)
    all     — All tests (default)
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root and skills to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'skills' / 'scripts'))

from config_constants import (
    EXPERT_HIDDEN_SIZE,
    EXPERT_NUM_HEADS,
    EXPERT_NUM_KV_HEADS,
    EXPERT_HEAD_DIM,
    EXPERT_NUM_LAYERS,
    EXPERT_SELF_ATTN_EVERY_N_LAYERS,
    EXPERT_NUM_CROSS_ATTN_LAYERS,
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    NUM_CONDITIONING_TOKENS,
    CONDITIONING_HIDDEN_SIZE,
    TIMESTEP_EMBED_DIM,
    MIN_PERIOD,
    MAX_PERIOD,
    NUM_DENOISING_STEPS,
    BATCH_SIZE,
    VLM_KV_DIM,
)

from action_expert_block import (
    NeuronSmolVLADenoisingWrapper,
    NeuronSmolVLAActionHead,
    SmolVLADenoisingStepRef,
    SmolVLAActionHeadConfig,
    get_weight_mapping,
)

from block_testing_utils import (
    test_action_head_cpu_interface,
    test_denoising_loop_correctness,
    _create_action_head_config,
    _make_denoiser_inputs,
)


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def make_smolvla_cfg(num_denoising_steps=NUM_DENOISING_STEPS):
    return _create_action_head_config(
        batch_size=BATCH_SIZE,
        action_chunk_size=ACTION_CHUNK_SIZE,
        action_dim=ACTION_DIM,
        num_conditioning_tokens=NUM_CONDITIONING_TOKENS,
        conditioning_hidden_size=CONDITIONING_HIDDEN_SIZE,
        timestep_embed_dim=TIMESTEP_EMBED_DIM,
        num_denoising_steps=num_denoising_steps,
    )


# ---------------------------------------------------------------------------
# Test 1: CPU interface smoke test
# ---------------------------------------------------------------------------

def test_cpu_interface():
    print("\n" + "=" * 80)
    print("TEST 1: CPU Interface Smoke Test")
    print("=" * 80)

    cfg = make_smolvla_cfg()

    # test_action_head_cpu_interface expects an action_head_class and init_kwargs
    # NeuronSmolVLAActionHead() works with no args (creates default config)
    test_action_head_cpu_interface(
        action_head_class=NeuronSmolVLAActionHead,
        action_head_config=cfg,
        init_kwargs={},
        verbose=True,
    )
    print("\n✓ TEST 1 PASSED: CPU interface smoke test")


# ---------------------------------------------------------------------------
# Test 2: Single denoising step correctness (direct comparison)
# ---------------------------------------------------------------------------

def test_single_step(seed: int = 42, atol: float = 1e-3):
    """
    Tests that NeuronSmolVLADenoisingWrapper produces identical output to
    SmolVLADenoisingStepRef (the reference implementation with same architecture),
    after weight synchronization.

    Strategy:
      1. Instantiate both wrapper and reference with the SAME random weights.
      2. Generate identical test inputs (noisy_actions, conditioning, timestep_emb, mask).
      3. Compare outputs with atol tolerance.

    Note: test_action_head_correctness() in block_testing_utils.py is designed for
    blocks with a SINGLE tensor input and cannot be used directly for 4-input models.
    We use a direct comparison approach here, which is equivalent but more appropriate
    for the multi-input signature.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Single Denoising Step Correctness")
    print(f"  action_chunk_size={ACTION_CHUNK_SIZE}, action_dim={ACTION_DIM}")
    print(f"  num_conditioning_tokens={NUM_CONDITIONING_TOKENS}")
    print(f"  atol={atol}")
    print("=" * 80)

    cfg = make_smolvla_cfg()
    inputs = _make_denoiser_inputs(cfg, seed=seed, zeros=False)
    noisy_actions, conditioning_tokens, timestep_embedding, attention_mask = inputs

    # 1. Build reference with random weights
    torch.manual_seed(seed)
    ref = SmolVLADenoisingStepRef().bfloat16().eval()
    torch.manual_seed(seed)
    for name, param in ref.named_parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)

    # 2. Build neuron wrapper with identical weights
    neuron = NeuronSmolVLADenoisingWrapper(config=None).bfloat16().eval()
    # Synchronize weights: both have identical architecture so state_dict keys match
    neuron.load_state_dict(ref.state_dict())

    print(f"\n  Weights synced: {len(list(ref.parameters()))} parameter tensors")

    # 3. Run both
    with torch.no_grad():
        ref_out = ref(noisy_actions, conditioning_tokens, timestep_embedding, attention_mask)
        neuron_out = neuron(noisy_actions, conditioning_tokens, timestep_embedding, attention_mask)

    # 4. Compare
    ref_fp32 = ref_out.float()
    neuron_fp32 = neuron_out.float()

    max_diff = (ref_fp32 - neuron_fp32).abs().max().item()
    mean_diff = (ref_fp32 - neuron_fp32).abs().mean().item()

    print(f"\n  Reference output:  shape={ref_fp32.shape}, mean={ref_fp32.mean().item():.6f}")
    print(f"  Neuron output:     shape={neuron_fp32.shape}, mean={neuron_fp32.mean().item():.6f}")
    print(f"\n  Max abs diff:  {max_diff:.8f}  (atol={atol})")
    print(f"  Mean abs diff: {mean_diff:.8f}")

    # Check for NaN/Inf
    assert not torch.isnan(neuron_fp32).any(), "NaN in Neuron output"
    assert not torch.isinf(neuron_fp32).any(), "Inf in Neuron output"
    assert not torch.isnan(ref_fp32).any(), "NaN in reference output"

    # Check shapes match
    assert ref_fp32.shape == neuron_fp32.shape, (
        f"Shape mismatch: ref={ref_fp32.shape} vs neuron={neuron_fp32.shape}"
    )
    assert ref_fp32.shape == (BATCH_SIZE, ACTION_CHUNK_SIZE, ACTION_DIM), (
        f"Wrong output shape: {ref_fp32.shape}, expected ({BATCH_SIZE}, {ACTION_CHUNK_SIZE}, {ACTION_DIM})"
    )

    # Check correctness within tolerance
    assert max_diff <= atol, (
        f"Single-step max diff {max_diff:.6f} exceeds atol={atol}.\n"
        f"Mean diff: {mean_diff:.6f}\n"
        f"This indicates a bug in the forward pass (not just numerical precision)."
    )

    print(f"\n{'=' * 80}")
    print(f"✓ TEST 2 PASSED: Single step matches reference (max_diff={max_diff:.2e} <= atol={atol})")
    print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# Test 3: Full denoising loop correctness
# ---------------------------------------------------------------------------

def test_denoising_loop(seed: int = 42, atol: float = 1e-2):
    """
    Tests that the full N-step denoising loop produces output matching
    the PyTorch reference, using test_denoising_loop_correctness() from
    block_testing_utils.py.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Full Denoising Loop Correctness")
    print(f"  num_steps={NUM_DENOISING_STEPS}, atol={atol}")
    print("=" * 80)

    cfg = make_smolvla_cfg(num_denoising_steps=NUM_DENOISING_STEPS)

    # Build reference with reproducible random weights
    torch.manual_seed(seed)
    ref = SmolVLADenoisingStepRef().bfloat16().eval()
    torch.manual_seed(seed)
    for name, param in ref.named_parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)

    # Build neuron wrapper with same weights
    neuron = NeuronSmolVLADenoisingWrapper(config=None).bfloat16().eval()
    neuron.load_state_dict(ref.state_dict())

    # Build action head and wire it up
    action_head = NeuronSmolVLAActionHead()
    action_head.denoising_wrapper = neuron
    action_head.attention_mask = action_head._build_attention_mask()

    def neuron_generate_actions_fn(conditioning_tokens, num_steps):
        """Denoising loop using NeuronSmolVLADenoisingWrapper."""
        B = conditioning_tokens.shape[0]
        torch.manual_seed(7)  # fixed seed for initial noise
        x_t = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, dtype=torch.bfloat16)
        timesteps = action_head._get_timestep_sequence(num_steps)
        dt = -1.0 / num_steps
        mask = action_head.attention_mask
        with torch.no_grad():
            for t in timesteps:
                temb = action_head._embed_timestep(t)
                v_t = action_head.denoising_wrapper(x_t, conditioning_tokens, temb, mask)
                x_t = x_t + dt * v_t.to(x_t.dtype)
        return x_t

    def pytorch_generate_actions_fn(conditioning_tokens, num_steps):
        """Reference denoising loop using SmolVLADenoisingStepRef."""
        B = conditioning_tokens.shape[0]
        torch.manual_seed(7)  # same fixed seed
        x_t = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, dtype=torch.bfloat16)
        timesteps = [1.0 - i / num_steps for i in range(num_steps)]
        dt = -1.0 / num_steps
        mask = torch.ones(B, 1, ACTION_CHUNK_SIZE, NUM_CONDITIONING_TOKENS, dtype=torch.int32)
        with torch.no_grad():
            for t in timesteps:
                from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
                cpu = torch.device("cpu")
                t_tensor = torch.tensor([t], dtype=torch.float32, device=cpu)
                temb = create_sinusoidal_pos_embedding(
                    t_tensor, TIMESTEP_EMBED_DIM, MIN_PERIOD, MAX_PERIOD, device=cpu
                ).to(torch.bfloat16).expand(B, -1)
                v_t = ref(x_t, conditioning_tokens, temb, mask)
                x_t = x_t + dt * v_t.to(x_t.dtype)
        return x_t

    test_denoising_loop_correctness(
        neuron_generate_actions_fn=neuron_generate_actions_fn,
        pytorch_generate_actions_fn=pytorch_generate_actions_fn,
        action_head_config=cfg,
        seed=seed,
        atol=atol,
        verbose=True,
    )
    print(f"\n✓ TEST 3 PASSED: Denoising loop correctness ({NUM_DENOISING_STEPS} steps, atol={atol})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmolVLA action expert tests")
    parser.add_argument("--test", choices=["cpu", "single", "loop", "all"], default="all")
    args = parser.parse_args()

    results = {}

    if args.test in ("cpu", "all"):
        try:
            test_cpu_interface()
            results["cpu"] = "PASS"
        except Exception as e:
            import traceback
            results["cpu"] = f"FAIL: {e}"
            traceback.print_exc()

    if args.test in ("single", "all"):
        try:
            test_single_step(atol=1e-3)
            results["single"] = "PASS"
        except Exception as e:
            import traceback
            results["single"] = f"FAIL: {e}"
            traceback.print_exc()

    if args.test in ("loop", "all"):
        try:
            test_denoising_loop(atol=1e-2)
            results["loop"] = "PASS"
        except Exception as e:
            import traceback
            results["loop"] = f"FAIL: {e}"
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")

    failed = [k for k, v in results.items() if v != "PASS"]
    if failed:
        print(f"\n{len(failed)} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} test(s) PASSED")
