"""
smolp3.py  –  SmolVLA inference on AWS Trainium

Loads pre-prepared inputs from artifacts/smolvla_fixed_sample.pt,
compiles the model with torch_neuronx, validates numerics against the
CPU reference, and benchmarks.

Requirements (Trainium box):
    pip install lerobot torch-neuronx neuronx-cc

Usage:
    python smolp3.py                 # compile + benchmark
    python smolp3.py --cached        # skip compile, load smolvla_neuron.pt
    python smolp3.py --bf16          # cast to bfloat16 before tracing
    python smolp3.py --cores 2       # DataParallel across 2 NeuronCores
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch_neuronx
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID      = "lerobot/smolvla_base"
SAMPLE_PATH   = "artifacts/smolvla_fixed_sample.pt"
COMPILED_PATH = "smolvla_neuron.pt"


# ---------------------------------------------------------------------------
# Wrapper: Neuron trace requires a flat positional-arg signature.
# ---------------------------------------------------------------------------
class SmolVLANeuronWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        image0:      torch.Tensor,   # (B, C, H, W)
        image1:      torch.Tensor,
        image2:      torch.Tensor,
        img_masks:   torch.Tensor,   # (B, N) bool
        lang_tokens: torch.Tensor,   # (B, L) int64
        lang_masks:  torch.Tensor,   # (B, L) int64
        state:       torch.Tensor,   # (B, D)
    ) -> torch.Tensor:
        return self.model.sample_actions(
            [image0, image1, image2], img_masks, lang_tokens, lang_masks, state
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_inputs(dtype: torch.dtype) -> tuple[tuple, torch.Tensor, int]:
    """Returns (inputs_tuple, ref_output, action_dim)."""
    sample = torch.load(SAMPLE_PATH, map_location="cpu")

    def f(t):  # cast floats, leave ints alone
        return t.to(dtype) if isinstance(t, torch.Tensor) and t.is_floating_point() else t

    img_masks = sample["img_masks"]
    if isinstance(img_masks, list):
        img_masks = torch.stack([f(m) for m in img_masks])
    else:
        img_masks = f(img_masks)

    inputs = (
        f(sample["image0"]),
        f(sample["image1"]),
        f(sample["image2"]),
        img_masks,
        sample["lang_tokens"],
        sample["lang_masks"],
        f(sample["state"]),
    )
    return inputs, sample["raw_out"], int(sample["original_action_dim"])


def compare(ref: torch.Tensor, out: torch.Tensor, label: str) -> None:
    ref  = ref.detach().float().cpu()
    out  = out.detach().float().cpu()
    diff = (ref - out).abs()
    ok   = torch.allclose(ref, out, atol=1e-2, rtol=1e-2)
    print(f"  [{label}]  max={diff.max():.5f}  mean={diff.mean():.5f}  "
          f"allclose={'PASS' if ok else 'FAIL'}")


def benchmark(fn, inputs: tuple, warmup: int, runs: int, label: str) -> None:
    with torch.inference_mode():
        for _ in range(warmup):
            fn(*inputs)
    times = []
    with torch.inference_mode():
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(*inputs)
            times.append(time.perf_counter() - t0)
    ms = [t * 1000 for t in times]
    print(f"  [{label}]  avg={sum(ms)/len(ms):.1f}ms  "
          f"min={min(ms):.1f}ms  max={max(ms):.1f}ms  (n={runs})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--bf16",   action="store_true")
    parser.add_argument("--cores",  type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs",   type=int, default=20)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # ── Inputs ───────────────────────────────────────────────────────────────
    print(f"Loading inputs from {SAMPLE_PATH}…")
    inputs, ref_out, action_dim = load_inputs(dtype)
    for i, t in enumerate(inputs):
        print(f"  inputs[{i}]: {tuple(t.shape)} {t.dtype}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID}…")
    policy  = SmolVLAPolicy.from_pretrained(MODEL_ID).eval()
    wrapper = SmolVLANeuronWrapper(policy.model).eval()
    if args.bf16:
        wrapper = wrapper.to(torch.bfloat16)

    # ── CPU sanity ────────────────────────────────────────────────────────────
    print("\nCPU sanity check…")
    with torch.inference_mode():
        cpu_out = wrapper(*inputs)
    compare(ref_out, cpu_out, "cpu vs saved-ref")

    # ── Compile or load ───────────────────────────────────────────────────────
    if args.cached and os.path.exists(COMPILED_PATH):
        print(f"\nLoading compiled model from {COMPILED_PATH}…")
        compiled = torch.jit.load(COMPILED_PATH)
    else:
        os.environ["NEURON_CC_FLAGS"] = (
            "--model-type=transformer --enable-fast-loading-neuron-binaries"
        )
        print("\nCompiling (this takes several minutes)…")
        compiled = torch_neuronx.trace(wrapper, inputs)
        if args.cores > 1:
            compiled = torch_neuronx.DataParallel(compiled, list(range(args.cores)))
        torch.jit.save(compiled, COMPILED_PATH)
        print(f"Saved to {COMPILED_PATH}")

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\nValidating…")
    with torch.inference_mode():
        neuron_out = compiled(*inputs)
    compare(cpu_out, neuron_out, "neuron vs cpu")
    print(f"  first_action (neuron): {neuron_out[:, 0, :action_dim].float().cpu()}")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print("\nBenchmark…")
    benchmark(compiled, inputs, warmup=args.warmup, runs=args.runs, label="neuron")


if __name__ == "__main__":
    main()
