"""
benchmark.py — GR00T N1.7 latency benchmark on AWS Trainium

Imports load_model from run_inference. Runs 70 iterations, skips first 20 as warmup.
Reports mean, median, p95 latency and throughput.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python benchmark.py
"""

import os
import sys
import time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'skills', 'scripts'))
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

from run_inference import load_model
from vlm_backbone_block import run_vit_cpu, make_backbone_inputs
from config_constants import (
    BATCH_SIZE, NUM_INFERENCE_TIMESTEPS, MAX_STATE_DIM, ACTION_HORIZON,
    MAX_ACTION_DIM, MODEL_PATH,
)

NUM_ITERATIONS = 70
NUM_WARMUP     = 20


def main():
    print("GR00T N1.7 Trainium Benchmark")
    print("=" * 50)
    print(f"Iterations: {NUM_ITERATIONS} ({NUM_WARMUP} warmup)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Denoising steps: {NUM_INFERENCE_TIMESTEPS}")
    print()

    # Load model
    model = load_model()
    hf_sd = model._hf_sd

    # Build inputs once (ViT preprocessing)
    print("Preparing inputs (ViT on CPU)...")
    backbone_in = make_backbone_inputs(B=BATCH_SIZE, n_text=16)
    inputs_embeds, pre_cos, pre_sin = run_vit_cpu(
        backbone_in['input_ids'], backbone_in['attention_mask'],
        backbone_in['pixel_values'], backbone_in['image_grid_thw'],
        hf_sd=hf_sd,
    )
    state = torch.zeros(BATCH_SIZE, 1, MAX_STATE_DIM, dtype=torch.bfloat16)
    embodiment_id = torch.zeros(BATCH_SIZE, dtype=torch.long)

    latencies = []

    print(f"Running {NUM_ITERATIONS} iterations...")
    for i in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        with torch.no_grad():
            actions = model.generate_actions(
                inputs_embeds=inputs_embeds,
                pre_cos=pre_cos,
                pre_sin=pre_sin,
                state=state,
                embodiment_id=embodiment_id,
                num_steps=NUM_INFERENCE_TIMESTEPS,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if i < NUM_WARMUP:
            print(f"  [warmup {i+1}/{NUM_WARMUP}] {elapsed_ms:.1f}ms")
        else:
            latencies.append(elapsed_ms)
            if (i - NUM_WARMUP) % 10 == 0:
                print(f"  [iter {i-NUM_WARMUP+1}/{NUM_ITERATIONS-NUM_WARMUP}] {elapsed_ms:.1f}ms")

    latencies = np.array(latencies)
    mean_ms   = latencies.mean()
    median_ms = np.median(latencies)
    p95_ms    = np.percentile(latencies, 95)
    throughput = 1000.0 / mean_ms  # inferences per second

    print()
    print("=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Mean latency:   {mean_ms:.2f} ms")
    print(f"Median latency: {median_ms:.2f} ms")
    print(f"P95 latency:    {p95_ms:.2f} ms")
    print(f"Throughput:     {throughput:.2f} inferences/sec")
    print(f"Action shape:   {actions.shape}")
    print()

    # Per-denoising-step estimate
    step_ms = mean_ms / NUM_INFERENCE_TIMESTEPS
    print(f"Estimated per-step: {step_ms:.2f} ms/step")
    print("=" * 50)


if __name__ == '__main__':
    main()
