"""
benchmark_subgraph.py — Per-subgraph latency breakdown for GR00T N1.7

Times each subgraph individually: ViT CPU, backbone NEFF, VL self-attn NEFF,
DiT NEFF (per denoising step), and total end-to-end.

Also times ViT compiled NEFF if compiled/vit/model.pt exists.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/groot-n1-port
    python benchmark_subgraph.py
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/home/ubuntu/groot-n1-port')
sys.path.insert(0, os.path.join('/home/ubuntu/groot-n1-port', 'skills', 'scripts'))
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

from run_inference import load_model
from vlm_backbone_block import run_vit_cpu, make_backbone_inputs
from config_constants import (
    BATCH_SIZE, NUM_INFERENCE_TIMESTEPS, MAX_STATE_DIM, MAX_ACTION_DIM,
    ACTION_HORIZON, DIT_INPUT_SEQ_LEN, NUM_CONDITIONING_TOKENS,
    NUM_TIMESTEP_BUCKETS, VIT_PATCH_SIZE, VIT_TEMPORAL_PATCH_SIZE,
    VISION_TOKENS_PER_IMAGE,
)

NUM_WARMUP = 10
NUM_MEASURE = 30

VIT_COMPILED = '/home/ubuntu/groot-n1-port/compiled/vit/model.pt'


def timeit(fn, warmup=NUM_WARMUP, measure=NUM_MEASURE):
    """Run fn() warmup+measure times, return array of measure latencies in ms."""
    for _ in range(warmup):
        fn()
    latencies = []
    for _ in range(measure):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return np.array(latencies)


def report(name, lats):
    print(f"  {name:40s}  mean={lats.mean():.2f}ms  p95={np.percentile(lats,95):.2f}ms")
    return lats.mean()


print("=" * 60)
print("GR00T N1.7 Per-Subgraph Latency Benchmark")
print(f"  Warmup={NUM_WARMUP}  Measure={NUM_MEASURE}  Batch={BATCH_SIZE}")
print("=" * 60)

# Load model
print("\nLoading model...")
model = load_model()
hf_sd = model._hf_sd

# Prepare inputs
print("Preparing inputs (ViT CPU once)...")
backbone_in = make_backbone_inputs(B=BATCH_SIZE, n_text=16)
with torch.no_grad():
    inputs_embeds, pre_cos, pre_sin = run_vit_cpu(
        backbone_in['input_ids'], backbone_in['attention_mask'],
        backbone_in['pixel_values'], backbone_in['image_grid_thw'],
        hf_sd=hf_sd,
    )
state = torch.zeros(BATCH_SIZE, 1, MAX_STATE_DIM, dtype=torch.bfloat16)
embodiment_id = torch.zeros(BATCH_SIZE, dtype=torch.long)
noisy_actions = torch.randn(BATCH_SIZE, ACTION_HORIZON, MAX_ACTION_DIM, dtype=torch.bfloat16)
cross_mask = torch.ones(BATCH_SIZE, 1, DIT_INPUT_SEQ_LEN, NUM_CONDITIONING_TOKENS, dtype=torch.int32)

# Pre-compute conditioning tokens (shared across DiT timing)
with torch.no_grad():
    backbone_out = model.backbone_wrapper(inputs_embeds, pre_cos, pre_sin)
    vlln_out = model.vlln(backbone_out)
    cond_tokens = model.vl_self_attn_wrapper(vlln_out)
    state_features = model.state_encoder(state.view(BATCH_SIZE, 1, -1).bfloat16(), embodiment_id)
    action_features = F.silu(model.action_encoder_W1(noisy_actions, embodiment_id))
    pos_ids = torch.arange(action_features.shape[1])
    action_features = action_features + model.position_embedding(pos_ids).unsqueeze(0)
    sa_embs = torch.cat([state_features, action_features], dim=1)
    t_bucket = torch.tensor([500], dtype=torch.long).expand(BATCH_SIZE)
    temb = model.timestep_encoder(t_bucket)

print("\n--- Per-Subgraph Latency ---")
means = {}

# 1. ViT CPU
pixel_values = backbone_in['pixel_values']
grid_thw = backbone_in['image_grid_thw']
lats = timeit(lambda: run_vit_cpu(
    backbone_in['input_ids'], backbone_in['attention_mask'],
    pixel_values, grid_thw, hf_sd=hf_sd,
))
means['vit_cpu'] = report("ViT (CPU)", lats)

# 2. ViT NEFF (if compiled)
if os.path.isfile(VIT_COMPILED):
    print("  Loading ViT NEFF...")
    import torch_neuronx
    vit_neff = torch.jit.load(VIT_COMPILED)
    # Create static pixel_values matching the ViT NEFF input
    npatch = 256
    pv_flat = torch.randn(
        npatch,
        3 * VIT_TEMPORAL_PATCH_SIZE * VIT_PATCH_SIZE * VIT_PATCH_SIZE,
        dtype=torch.bfloat16,
    )
    lats_neff = timeit(lambda: vit_neff(pv_flat))
    means['vit_neff'] = report("ViT (NEFF)", lats_neff)
else:
    print(f"  ViT NEFF not found at {VIT_COMPILED}, skipping NEFF timing")

# 3. Backbone NEFF
lats = timeit(lambda: model.backbone_wrapper(inputs_embeds, pre_cos, pre_sin))
means['backbone'] = report("Backbone NEFF", lats)

# 4. VL self-attn NEFF
lats = timeit(lambda: model.vl_self_attn_wrapper(vlln_out))
means['vl_self_attn'] = report("VL self-attn NEFF", lats)

# 5. DiT NEFF (single denoising step)
lats = timeit(lambda: model.dit_wrapper(sa_embs, cond_tokens, temb, cross_mask))
means['dit_step'] = report("DiT NEFF (1 step)", lats)

# 6. Total end-to-end (without ViT)
def run_full(inputs_embeds=inputs_embeds, pre_cos=pre_cos, pre_sin=pre_sin,
             state=state, embodiment_id=embodiment_id):
    with torch.no_grad():
        return model.generate_actions(
            inputs_embeds=inputs_embeds,
            pre_cos=pre_cos,
            pre_sin=pre_sin,
            state=state,
            embodiment_id=embodiment_id,
        )

lats = timeit(run_full)
means['e2e_no_vit'] = report(f"E2E excl. ViT ({NUM_INFERENCE_TIMESTEPS} DiT steps)", lats)

# Total including ViT CPU
lats_full = timeit(lambda: (
    run_vit_cpu(
        backbone_in['input_ids'], backbone_in['attention_mask'],
        pixel_values, grid_thw, hf_sd=hf_sd,
    ),
    run_full(
        *run_vit_cpu(
            backbone_in['input_ids'], backbone_in['attention_mask'],
            pixel_values, grid_thw, hf_sd=hf_sd,
        ),
        state, embodiment_id,
    )
))
means['e2e_with_vit_cpu'] = report("E2E incl. ViT CPU", lats_full)

print("\n--- Summary ---")
print(f"  ViT on CPU:         {means['vit_cpu']:.2f} ms")
if 'vit_neff' in means:
    print(f"  ViT on NEFF:        {means['vit_neff']:.2f} ms")
print(f"  Backbone NEFF:      {means['backbone']:.2f} ms")
print(f"  VL self-attn NEFF:  {means['vl_self_attn']:.2f} ms")
print(f"  DiT (per step):     {means['dit_step']:.2f} ms")
print(f"  DiT (x{NUM_INFERENCE_TIMESTEPS} steps):      {means['dit_step']*NUM_INFERENCE_TIMESTEPS:.2f} ms (estimated)")
print(f"  Total excl. ViT:    {means['e2e_no_vit']:.2f} ms")
print(f"  Total incl. ViT:    {means['e2e_with_vit_cpu']:.2f} ms")
print(f"  Throughput:         {1000/means['e2e_no_vit']:.1f} inferences/sec (excl. ViT)")
