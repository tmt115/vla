"""
Numerical correctness test for SmolVLA compiled NEFFs.

For each subgraph, runs the same inputs through:
  1. CPU reference (our reimplemented nn.Module with HF weights, bfloat16)
  2. Compiled NEFF (Trainium)

Compares outputs with tolerances appropriate for bfloat16 execution.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/smol-port-skilled
    python -m pytest tests/test_correctness.py -v -s
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')
sys.path.insert(0, str(ROOT / 'skills' / 'scripts'))

import torch
import torch_neuronx  # must come before torch.jit.load
from safetensors.torch import load_file

from config_constants import (
    BATCH_SIZE, VISION_IMAGE_SIZE, HIDDEN_SIZE, PREFIX_SEQ_LEN,
    NUM_TEXT_TOKENS, NUM_STATE_TOKENS, ACTION_CHUNK_SIZE, ACTION_DIM,
    NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE,
    TIMESTEP_EMBED_DIM, MIN_PERIOD, MAX_PERIOD,
)
from modeling_smolvla_neuron import (
    NeuronSmolVLAVisionModel,  NeuronSmolVLAVisionHead,
    NeuronSmolVLABackboneModel, NeuronSmolVLABackboneHead,
)
from action_expert_block import (
    NeuronSmolVLADenoisingWrapper, SmolVLAActionHeadConfig,
    get_hf_to_neuron_weight_mapping,
)

CHECKPOINT   = ROOT / 'checkpoint' / 'model.safetensors'
COMPILED_DIR = ROOT / 'compiled'

def _hf_sd():
    return load_file(str(CHECKPOINT))


def _t_emb(t: float) -> torch.Tensor:
    from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
    t_tensor = torch.tensor([t], dtype=torch.float32)
    emb = create_sinusoidal_pos_embedding(
        t_tensor, TIMESTEP_EMBED_DIM, MIN_PERIOD, MAX_PERIOD, device=torch.device('cpu')
    )
    return emb.to(torch.bfloat16).expand(BATCH_SIZE, -1)


def _report(name, cpu_out, neff_out, max_mean_diff, min_cosine_sim):
    """
    Correctness check: mean absolute diff + cosine similarity.

    mean absolute diff catches wrong implementations (e.g. uninitialized weights).
    cosine similarity catches sign flips or scale errors that mean diff misses.
    Relative diff (|diff|/|ref|) is avoided — near-zero outputs inflate it
    artificially even when the outputs are numerically identical.

    max_mean_diff:    threshold on mean(|cpu - neff|)
    min_cosine_sim:   threshold on cosine_similarity(cpu.flatten(), neff.flatten())
    """
    diff = (cpu_out.float() - neff_out.float()).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()
    a = cpu_out.float().flatten()
    b = neff_out.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    print(f"  {name}: max_diff={max_diff:.4f}  mean_diff={mean_diff:.4f}"
          f"  cosine_sim={cos_sim:.6f}")
    assert mean_diff <= max_mean_diff, (
        f"{name}: mean absolute diff {mean_diff:.4f} exceeds threshold {max_mean_diff}"
    )
    assert cos_sim >= min_cosine_sim, (
        f"{name}: cosine similarity {cos_sim:.6f} below threshold {min_cosine_sim}"
    )


# ---------------------------------------------------------------------------
# Test 1: Vision encoder
# ---------------------------------------------------------------------------

def test_vision_correctness():
    print("\n── Vision encoder correctness ──")
    hf_sd = _hf_sd()

    # CPU reference
    cpu_model = NeuronSmolVLAVisionModel()
    vis_sd = NeuronSmolVLAVisionHead.convert_hf_to_neuron_state_dict(hf_sd, None)
    cpu_model.load_state_dict(vis_sd, strict=True)
    cpu_model = cpu_model.bfloat16().eval()

    # NEFF
    neff = torch.jit.load(str(COMPILED_DIR / 'vision' / 'vision_encoder.pt'))

    torch.manual_seed(42)
    pixel_values = torch.randn(BATCH_SIZE, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE,
                               dtype=torch.bfloat16)

    with torch.no_grad():
        cpu_out  = cpu_model(pixel_values)
        neff_out = neff(pixel_values)

    print(f"  shapes: cpu={tuple(cpu_out.shape)}  neff={tuple(neff_out.shape)}")
    _report("vision_encoder", cpu_out, neff_out.to(cpu_out.dtype),
            max_mean_diff=0.1, min_cosine_sim=0.9997)
    print("  PASS ✓")


# ---------------------------------------------------------------------------
# Test 2: VLM backbone
# ---------------------------------------------------------------------------

def test_vlm_correctness():
    print("\n── VLM backbone correctness ──")
    hf_sd = _hf_sd()

    # CPU reference
    cpu_model = NeuronSmolVLABackboneModel()
    bac_sd = NeuronSmolVLABackboneHead.convert_hf_to_neuron_state_dict(hf_sd, None)
    missing, unexpected = cpu_model.load_state_dict(bac_sd, strict=False)
    non_rope = [k for k in missing if 'rope_' not in k]
    assert not non_rope, f"Missing non-rope keys: {non_rope}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    cpu_model = cpu_model.bfloat16().eval()

    # NEFF
    neff = torch.jit.load(str(COMPILED_DIR / 'vlm' / 'vlm_backbone.pt'))

    torch.manual_seed(42)
    B = BATCH_SIZE
    prefix   = torch.randn(B, PREFIX_SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)
    attn_mask = torch.ones(B, PREFIX_SEQ_LEN, PREFIX_SEQ_LEN, dtype=torch.bool)
    pos_ids   = torch.arange(PREFIX_SEQ_LEN).unsqueeze(0).expand(B, -1).clone()

    with torch.no_grad():
        cpu_out  = cpu_model(prefix, attn_mask, pos_ids)
        neff_out = neff(prefix, attn_mask, pos_ids)

    print(f"  shapes: cpu={tuple(cpu_out.shape)}  neff={tuple(neff_out.shape)}")
    _report("vlm_backbone", cpu_out, neff_out.to(cpu_out.dtype),
            max_mean_diff=0.1, min_cosine_sim=0.997)
    print("  PASS ✓")


# ---------------------------------------------------------------------------
# Test 3: Action expert — single step
# ---------------------------------------------------------------------------

def test_expert_correctness():
    print("\n── Action expert correctness (single step) ──")
    hf_sd = _hf_sd()

    # CPU reference
    cpu_model = NeuronSmolVLADenoisingWrapper().bfloat16().eval()
    mapping   = get_hf_to_neuron_weight_mapping()
    expert_sd = {v: hf_sd[k] for k, v in mapping.items() if k in hf_sd}
    missing, unexpected = cpu_model.load_state_dict(expert_sd, strict=False)
    non_buf = [k for k in missing if not any(s in k for s in
               ('rope_sin', 'rope_cos', 'self_attn_mask'))]
    assert not non_buf, f"Missing non-buffer keys: {non_buf}"
    assert not unexpected, f"Unexpected keys: {unexpected}"

    # NEFF
    neff = torch.jit.load(str(COMPILED_DIR / 'action_expert' / 'action_expert.pt'))

    torch.manual_seed(42)
    B = BATCH_SIZE
    x_t        = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, dtype=torch.bfloat16)
    cond       = torch.randn(B, NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE,
                             dtype=torch.bfloat16)
    t_emb      = _t_emb(0.5)
    cross_mask = torch.ones(B, 1, ACTION_CHUNK_SIZE, NUM_CONDITIONING_TOKENS,
                            dtype=torch.int32)

    with torch.no_grad():
        cpu_out  = cpu_model(x_t, cond, t_emb, cross_mask)
        neff_out = neff(x_t, cond, t_emb, cross_mask)

    print(f"  shapes: cpu={tuple(cpu_out.shape)}  neff={tuple(neff_out.shape)}")
    _report("action_expert_step", cpu_out, neff_out.to(cpu_out.dtype),
            max_mean_diff=0.15, min_cosine_sim=0.9999)
    print("  PASS ✓")


if __name__ == '__main__':
    test_vision_correctness()
    test_vlm_correctness()
    test_expert_correctness()
    print("\n" + "=" * 60)
    print("ALL CORRECTNESS CHECKS PASSED")
    print("=" * 60)
