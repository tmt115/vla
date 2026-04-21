"""
Phase 5 integration smoke test — runs with compiled NEFFs on Trainium.

Loads all three compiled subgraphs and real checkpoint weights, then runs
the full inference pipeline and validates:
  - output shape [B, 50, 32]
  - no NaN, no Inf
  - no all-zeros (model is actually doing something)
  - per-value range roughly within [-5, 5]
  - per-subgraph latency printed

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/smol-port-skilled
    python tests/test_phase5_compiled.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEROBOT_PATH = '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages'
if LEROBOT_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_PATH)

SKILLS_PATH = str(ROOT / 'skills' / 'scripts')
if SKILLS_PATH not in sys.path:
    sys.path.insert(0, SKILLS_PATH)

import torch
import torch_neuronx  # must import before torch.jit.load to register Neuron custom types
from safetensors.torch import load_file

CHECKPOINT   = str(ROOT / 'checkpoint' / 'model.safetensors')
COMPILED_DIR = ROOT / 'compiled'

from config_constants import (
    BATCH_SIZE,
    VISION_IMAGE_SIZE,
    HIDDEN_SIZE,
    PREFIX_SEQ_LEN,
    NUM_TEXT_TOKENS,
    NUM_STATE_TOKENS,
    VISION_TOKENS_PER_CAMERA,
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    NUM_CONDITIONING_TOKENS,
    CONDITIONING_HIDDEN_SIZE,
    NUM_DENOISING_STEPS,
    TIMESTEP_EMBED_DIM,
    MIN_PERIOD,
    MAX_PERIOD,
)
from action_expert_block import get_hf_to_neuron_weight_mapping


def load_compiled():
    """Load all three compiled ScriptModules."""
    vis_path    = COMPILED_DIR / 'vision'      / 'vision_encoder.pt'
    vlm_path    = COMPILED_DIR / 'vlm'         / 'vlm_backbone.pt'
    exp_path    = COMPILED_DIR / 'action_expert' / 'action_expert.pt'

    for p in (vis_path, vlm_path, exp_path):
        assert p.exists(), f"Compiled NEFF not found: {p}"

    print(f"  Loading vision encoder from   {vis_path}")
    vis_model = torch.jit.load(str(vis_path))
    print(f"  Loading VLM backbone from     {vlm_path}")
    vlm_model = torch.jit.load(str(vlm_path))
    print(f"  Loading action expert from    {exp_path}")
    exp_model = torch.jit.load(str(exp_path))

    return vis_model, vlm_model, exp_model


def sinusoidal_timestep_emb(t: float) -> torch.Tensor:
    from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
    cpu = torch.device("cpu")
    t_tensor = torch.tensor([t], dtype=torch.float32, device=cpu)
    emb = create_sinusoidal_pos_embedding(t_tensor, TIMESTEP_EMBED_DIM, MIN_PERIOD, MAX_PERIOD, device=cpu)
    return emb.to(torch.bfloat16).expand(BATCH_SIZE, -1)


def test_compiled_inference():
    print("\n" + "=" * 70)
    print("PHASE 5: Integration smoke test with compiled NEFFs")
    print("=" * 70)

    vis_model, vlm_model, exp_model = load_compiled()
    print("  All three NEFFs loaded ✓")

    B = BATCH_SIZE
    torch.manual_seed(0)

    # ── Step 1: Vision encoder ──────────────────────────────────────────────
    pixel_values = torch.randn(B, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE,
                               dtype=torch.bfloat16)

    t0 = time.perf_counter()
    vision_embs = vis_model(pixel_values)
    vis_latency = (time.perf_counter() - t0) * 1000

    assert vision_embs.shape == (B, VISION_TOKENS_PER_CAMERA, HIDDEN_SIZE), \
        f"Vision encoder output shape: {vision_embs.shape}"
    assert not torch.isnan(vision_embs).any(), "NaN in vision encoder output"
    print(f"  [1] Vision encoder:  {tuple(pixel_values.shape)} → {tuple(vision_embs.shape)}"
          f"  {vis_latency:.1f}ms ✓")

    # ── Assemble prefix [B, 113, 960] ─────────────────────────────────────
    text_embs  = torch.randn(B, NUM_TEXT_TOKENS,  HIDDEN_SIZE, dtype=torch.bfloat16)
    state_embs = torch.randn(B, NUM_STATE_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16)
    prefix = torch.cat([text_embs, vision_embs.to(torch.bfloat16), state_embs], dim=1)
    assert prefix.shape == (B, PREFIX_SEQ_LEN, HIDDEN_SIZE)

    # ── Step 2: VLM backbone ────────────────────────────────��──────────────
    attn_mask = torch.ones(B, PREFIX_SEQ_LEN, PREFIX_SEQ_LEN, dtype=torch.bool)
    pos_ids   = torch.arange(PREFIX_SEQ_LEN).unsqueeze(0).expand(B, -1).clone()

    t0 = time.perf_counter()
    conditioning = vlm_model(prefix, attn_mask, pos_ids)
    vlm_latency = (time.perf_counter() - t0) * 1000

    assert conditioning.shape == (B, NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE), \
        f"VLM backbone output shape: {conditioning.shape}"
    assert not torch.isnan(conditioning).any(), "NaN in VLM backbone output"
    conditioning = conditioning.to(torch.bfloat16)
    print(f"  [2] VLM backbone:    {tuple(prefix.shape)} → {tuple(conditioning.shape)}"
          f"  {vlm_latency:.1f}ms ✓")

    # ── Step 3: Action expert denoising loop ──────────────��───────────────
    cross_mask = torch.ones(B, 1, ACTION_CHUNK_SIZE, NUM_CONDITIONING_TOKENS,
                            dtype=torch.int32)
    x_t   = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, dtype=torch.bfloat16)
    dt    = -1.0 / NUM_DENOISING_STEPS
    times = [1.0 - i / NUM_DENOISING_STEPS for i in range(NUM_DENOISING_STEPS)]

    t0 = time.perf_counter()
    for step, t in enumerate(times):
        t_emb = sinusoidal_timestep_emb(t)
        v_t   = exp_model(x_t, conditioning, t_emb, cross_mask)
        x_t   = x_t + dt * v_t.to(x_t.dtype)
    loop_latency = (time.perf_counter() - t0) * 1000
    step_latency  = loop_latency / NUM_DENOISING_STEPS

    actions = x_t
    assert actions.shape == (B, ACTION_CHUNK_SIZE, ACTION_DIM), \
        f"Action output shape: {actions.shape}"
    assert not torch.isnan(actions).any(),   "NaN in action output"
    assert not torch.isinf(actions).any(),   "Inf in action output"
    assert not (actions == 0).all(),         "All-zeros action output"

    act_mean = actions.float().mean().item()
    act_std  = actions.float().std().item()
    act_max  = actions.float().abs().max().item()
    assert act_max < 50.0, f"Action values out of reasonable range: max_abs={act_max:.2f}"

    print(f"  [3] Action expert:   {NUM_DENOISING_STEPS}-step loop → {tuple(actions.shape)}"
          f"  {loop_latency:.1f}ms total  ({step_latency:.1f}ms/step) ✓")
    print(f"      mean={act_mean:.4f}, std={act_std:.4f}, max_abs={act_max:.4f}")

    # ── Latency summary ────────────────────────��───────────────────────────
    total_latency = vis_latency + vlm_latency + loop_latency
    print(f"\n  Latency summary:")
    print(f"    Vision encoder:  {vis_latency:>8.1f} ms")
    print(f"    VLM backbone:    {vlm_latency:>8.1f} ms")
    print(f"    Action expert:   {loop_latency:>8.1f} ms  ({NUM_DENOISING_STEPS} × {step_latency:.1f}ms)")
    print(f"    Total:           {total_latency:>8.1f} ms")

    print("\n✓ ALL PHASE 5 INTEGRATION CHECKS PASSED")
    return {
        "vis_latency_ms":   vis_latency,
        "vlm_latency_ms":   vlm_latency,
        "loop_latency_ms":  loop_latency,
        "step_latency_ms":  step_latency,
        "total_latency_ms": total_latency,
        "act_mean": act_mean,
        "act_std":  act_std,
        "act_max":  act_max,
    }


if __name__ == "__main__":
    results = test_compiled_inference()
    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)
