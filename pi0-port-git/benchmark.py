"""
benchmark.py — π0 Trainium latency benchmark.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python benchmark.py
"""

import sys, time, statistics
import torch
import numpy as np

sys.path.insert(0, '/home/ubuntu/pi0-port')
sys.path.insert(0, '/home/ubuntu/pi0-port/skills/scripts')

from run_inference import load_model
from config_constants import NUM_CAMERAS, MAX_LANG_TOKENS, ACTUAL_STATE_DIM, ACTUAL_ACTION_DIM, CHUNK_SIZE


def make_dummy_inputs():
    images = [torch.randn(1, 3, 224, 224, dtype=torch.float32) * 0.5 for _ in range(NUM_CAMERAS)]
    lang_tokens = torch.zeros(1, MAX_LANG_TOKENS, dtype=torch.long)
    lang_masks = torch.ones(1, MAX_LANG_TOKENS, dtype=torch.bool)
    state = torch.zeros(1, ACTUAL_STATE_DIM, dtype=torch.float32)
    return images, lang_tokens, lang_masks, state


def benchmark_subgraph(fn, inputs, name: str, warmup: int = 5, iters: int = 50):
    """Benchmark a single callable."""
    # Warmup
    for _ in range(warmup):
        fn(*inputs)

    timings = []
    for _ in range(iters):
        start = time.perf_counter()
        fn(*inputs)
        timings.append((time.perf_counter() - start) * 1000)

    st = sorted(timings)
    print(f"{name}:")
    print(f"  mean={statistics.mean(timings):.1f}ms  "
          f"median={statistics.median(timings):.1f}ms  "
          f"p95={st[int(0.95 * len(st))]:.1f}ms  "
          f"std={statistics.stdev(timings):.1f}ms  "
          f"throughput={1000 / statistics.mean(timings):.2f} inf/sec")
    return timings


if __name__ == "__main__":
    print("=== π0 Trainium Benchmark ===\n")

    model = load_model()
    images, lang_tokens, lang_masks, state = make_dummy_inputs()

    print("Benchmarking per-subgraph latency...\n")

    # ── Vision encoder (per camera, 3× in pipeline) ────────────────────────
    vis_timings = benchmark_subgraph(
        model.vision_encoder,
        (images[0],),
        "Vision Encoder (1 camera)",
        warmup=3, iters=50,
    )

    # ── Prefix encoder ─────────────────────────────────────────────────────
    import torch.nn.functional as F
    import math
    from config_constants import VLM_HIDDEN_SIZE, SIGLIP_NUM_IMAGE_TOKENS
    with torch.no_grad():
        img_feats = [model.vision_encoder(img) * math.sqrt(VLM_HIDDEN_SIZE) for img in images]
        lang_embs = model.language_embedder(lang_tokens)
        prefix_embs = torch.cat([f.to(torch.bfloat16) for f in img_feats] + [lang_embs], dim=1)

    pre_timings = benchmark_subgraph(
        model.prefix_encoder,
        (prefix_embs,),
        "Prefix Encoder (Gemma 2B, 816 tokens)",
        warmup=3, iters=30,
    )

    # ── Full pipeline (end-to-end) ─────────────────────────────────────────
    def full_pipeline(*_):
        return model.generate_actions(images, lang_tokens, lang_masks, state)

    full_timings = benchmark_subgraph(
        full_pipeline,
        (),
        f"Full Pipeline (3 cams + 10-step denoising → {CHUNK_SIZE} actions)",
        warmup=3, iters=30,
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n=== Summary ===")
    print(f"Vision encoder (×3): {statistics.mean(vis_timings) * 3:.1f} ms")
    print(f"Prefix encoder:      {statistics.mean(pre_timings):.1f} ms")
    suffix_est = statistics.mean(full_timings) - statistics.mean(vis_timings) * 3 - statistics.mean(pre_timings)
    print(f"Suffix denoiser (×10 est.): {max(suffix_est, 0):.1f} ms")
    print(f"End-to-end:          {statistics.mean(full_timings):.1f} ms")
    print(f"Throughput:          {1000 / statistics.mean(full_timings):.2f} policy steps/sec")
