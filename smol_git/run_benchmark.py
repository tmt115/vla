"""
run_benchmark.py
================
End-to-end benchmark for the SmolVLA Trainium port.

Steps
-----
1. Load real smolvla_base checkpoint → convert to neuron state dicts
2. Compile NeuronSmolVLAPrefix and NeuronSmolVLADenoiseStep with torch_neuronx.trace
   (compilation is cached in /var/tmp/neuron-compile-cache; takes ~10 min first run)
3. Load weights into compiled models
4. Run WARMUP_RUNS forward passes (warm up NeuronCore)
5. Run BENCH_RUNS forward passes and report latency

Run on Trainium (trn1.2xl):
    PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \\
    python /home/ubuntu/smol_port/run_benchmark.py

Optional flags (env vars):
    SKIP_COMPILE=1   — skip recompilation, load previously compiled NEFFs from cache
    NUM_STEPS=10     — number of Euler denoising steps per inference call (default 10)
    BENCH_RUNS=50    — number of timed inference runs (default 50)
    WARMUP_RUNS=5    — number of warmup runs before timing (default 5)
"""

import os
import sys
import time
import torch

sys.path.insert(0, '/home/ubuntu/smol_port')
sys.path.insert(0, '/home/ubuntu/smol_port/tests')
import _transformers_compat  # bypass huggingface-hub check

from safetensors.torch import load_file
from weight_mapping import convert_hf_to_neuron_state_dicts
from modeling_smolvla_neuron import NeuronSmolVLAPrefix, NeuronSmolVLADenoiseStep

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT_PATH    = os.path.expanduser(
    "~/.cache/huggingface/hub/models--lerobot--smolvla_base/"
    "snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors"
)
COMPILED_DIR = "/home/ubuntu/smol_port/compiled"
NUM_STEPS    = int(os.environ.get("NUM_STEPS",    10))
BENCH_RUNS   = int(os.environ.get("BENCH_RUNS",   50))
WARMUP_RUNS  = int(os.environ.get("WARMUP_RUNS",   5))
SKIP_COMPILE = os.environ.get("SKIP_COMPILE", "0") == "1"

# Static shapes
B           = 1
PREFIX_LEN  = 241
SUFFIX_LEN  = 50
VLM_HIDDEN  = 960
EXP_HIDDEN  = 720
ACTION_DIM  = 32
NUM_LAYERS  = 16
KV_HEADS    = 5
HEAD_DIM    = 64

os.makedirs(COMPILED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Example inputs (static shapes — must match at trace and run time)
# ---------------------------------------------------------------------------
def make_prefix_inputs():
    prefix_embs = torch.zeros(B, PREFIX_LEN, VLM_HIDDEN, dtype=torch.bfloat16)
    attn_mask   = torch.ones(B, PREFIX_LEN, PREFIX_LEN, dtype=torch.bool)
    pos_ids     = torch.arange(PREFIX_LEN, dtype=torch.int64).unsqueeze(0)
    return prefix_embs, attn_mask, pos_ids


def make_denoise_inputs(kv_tensors):
    noisy   = torch.zeros(B, SUFFIX_LEN, ACTION_DIM, dtype=torch.float32)
    ts      = torch.tensor([0.5], dtype=torch.float32)
    return noisy, ts, *kv_tensors


# ---------------------------------------------------------------------------
# Step 1 — Load and convert weights
# ---------------------------------------------------------------------------
def load_weights():
    print(f"Loading checkpoint from {CKPT_PATH} ...")
    hf_sd = load_file(CKPT_PATH)
    print(f"  {len(hf_sd)} keys loaded")

    parts = convert_hf_to_neuron_state_dicts(hf_sd)
    print("  Weight mapping complete")
    return parts


# ---------------------------------------------------------------------------
# Step 2 — Compile (or load from cache)
# ---------------------------------------------------------------------------
def compile_models(parts):
    import torch_neuronx

    prefix_path  = os.path.join(COMPILED_DIR, "prefix.pt")
    denoise_path = os.path.join(COMPILED_DIR, "denoise.pt")

    if SKIP_COMPILE and os.path.exists(prefix_path) and os.path.exists(denoise_path):
        print("Loading pre-compiled NEFFs from disk ...")
        prefix_model  = torch.jit.load(prefix_path)
        denoise_model = torch.jit.load(denoise_path)
        return prefix_model, denoise_model

    # --- Compile NeuronSmolVLAPrefix ---
    print("\nCompiling NeuronSmolVLAPrefix (may take several minutes) ...")
    prefix_cpu = NeuronSmolVLAPrefix().to(torch.bfloat16)
    missing, unexpected = prefix_cpu.load_state_dict(parts["prefix"], strict=True)
    assert not missing and not unexpected, f"prefix load: missing={missing}, unexpected={unexpected}"

    prefix_inputs = make_prefix_inputs()
    t0 = time.time()
    prefix_model = torch_neuronx.trace(prefix_cpu, prefix_inputs)
    print(f"  Prefix compilation: {time.time()-t0:.1f}s")
    torch.jit.save(prefix_model, prefix_path)
    print(f"  Saved to {prefix_path}")

    # --- Run prefix once to get KV tensors for denoise trace ---
    kv_tensors = prefix_model(*prefix_inputs)

    # --- Compile NeuronSmolVLADenoiseStep ---
    print("\nCompiling NeuronSmolVLADenoiseStep (may take several minutes) ...")
    denoise_cpu = NeuronSmolVLADenoiseStep().to(torch.bfloat16)
    missing, unexpected = denoise_cpu.load_state_dict(parts["denoise_step"], strict=False)
    unexpected = [k for k in unexpected]
    real_missing = [k for k in missing if "sine_scale" not in k]
    assert not real_missing and not unexpected, \
        f"denoise load: missing={real_missing}, unexpected={unexpected}"

    noisy = torch.zeros(B, SUFFIX_LEN, ACTION_DIM, dtype=torch.float32)
    ts    = torch.tensor([0.5], dtype=torch.float32)
    denoise_inputs = (noisy, ts, *kv_tensors)

    t0 = time.time()
    denoise_model = torch_neuronx.trace(denoise_cpu, denoise_inputs)
    print(f"  Denoise compilation: {time.time()-t0:.1f}s")
    torch.jit.save(denoise_model, denoise_path)
    print(f"  Saved to {denoise_path}")

    return prefix_model, denoise_model


# ---------------------------------------------------------------------------
# Step 3 — Benchmark
# ---------------------------------------------------------------------------
def benchmark(prefix_model, denoise_model):
    prefix_inputs = make_prefix_inputs()

    print(f"\nBenchmarking: {WARMUP_RUNS} warmup + {BENCH_RUNS} timed runs, "
          f"{NUM_STEPS} Euler steps each")

    def run_one():
        # Prefix pass (once per inference)
        kv_tensors = prefix_model(*prefix_inputs)

        # Euler denoising loop (NUM_STEPS iterations)
        noisy = torch.randn(B, SUFFIX_LEN, ACTION_DIM, dtype=torch.float32)
        for step in range(NUM_STEPS):
            ts = torch.full((B,), step / NUM_STEPS, dtype=torch.float32)
            v_t = denoise_model(noisy, ts, *kv_tensors)
            noisy = noisy - (1.0 / NUM_STEPS) * v_t   # Euler step

        return noisy   # final action prediction [B, 50, 32]

    # Warmup
    for _ in range(WARMUP_RUNS):
        run_one()
    print("  Warmup done")

    # Timed runs
    latencies = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        result = run_one()
        latencies.append((time.perf_counter() - t0) * 1000)   # ms

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = sum(latencies) / len(latencies)

    print(f"\n{'='*50}")
    print(f"  Result shape: {tuple(result.shape)}  dtype={result.dtype}")
    print(f"  Latency per inference ({NUM_STEPS} Euler steps):")
    print(f"    Mean  : {mean:.1f} ms")
    print(f"    P50   : {p50:.1f} ms")
    print(f"    P99   : {p99:.1f} ms")
    print(f"    Min   : {latencies[0]:.1f} ms")
    print(f"    Max   : {latencies[-1]:.1f} ms")
    print(f"{'='*50}")

    # Per-step breakdown
    print("\nPer-step breakdown (estimate):")
    prefix_lats = []
    denoise_lats = []
    kv = prefix_model(*prefix_inputs)
    for _ in range(20):
        t0 = time.perf_counter()
        prefix_model(*prefix_inputs)
        prefix_lats.append((time.perf_counter() - t0) * 1000)

        noisy = torch.randn(B, SUFFIX_LEN, ACTION_DIM, dtype=torch.float32)
        ts    = torch.full((B,), 0.5, dtype=torch.float32)
        t0 = time.perf_counter()
        denoise_model(noisy, ts, *kv)
        denoise_lats.append((time.perf_counter() - t0) * 1000)

    print(f"  Prefix pass:       {sorted(prefix_lats)[10]:.1f} ms (p50)")
    print(f"  Single denoise step: {sorted(denoise_lats)[10]:.1f} ms (p50)")
    print(f"  Estimated total ({NUM_STEPS} steps): "
          f"{sorted(prefix_lats)[10] + NUM_STEPS*sorted(denoise_lats)[10]:.1f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parts = load_weights()
    prefix_model, denoise_model = compile_models(parts)
    benchmark(prefix_model, denoise_model)
