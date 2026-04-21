"""
benchmark.py — SmolVLA Trainium port latency benchmark

Measures wall-clock latency of each compiled subgraph and compares against
the HuggingFace CPU reference. Reports per-step action expert latency,
full inference latency, and throughput (inferences/sec).

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/smol-port-skilled
    python benchmark.py                    # default: 20 warmup + 50 timed runs
    python benchmark.py --runs 100         # more timed runs
    python benchmark.py --no-reference     # skip CPU reference (fast)
    python benchmark.py --steps 10         # denoising steps (default: NUM_DENOISING_STEPS)
"""

import argparse
import sys
import time
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')
sys.path.insert(0, str(ROOT / 'skills' / 'scripts'))

import torch
import torch_neuronx  # must come before torch.jit.load
from safetensors.torch import load_file

from config_constants import (
    BATCH_SIZE, VISION_IMAGE_SIZE, HIDDEN_SIZE, PREFIX_SEQ_LEN,
    NUM_TEXT_TOKENS, NUM_STATE_TOKENS, VISION_TOKENS_PER_CAMERA,
    ACTION_CHUNK_SIZE, ACTION_DIM, NUM_CONDITIONING_TOKENS,
    CONDITIONING_HIDDEN_SIZE, NUM_DENOISING_STEPS,
    TIMESTEP_EMBED_DIM, MIN_PERIOD, MAX_PERIOD,
)

COMPILED_DIR = ROOT / 'compiled'
CHECKPOINT   = ROOT / 'checkpoint' / 'model.safetensors'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sinusoidal_emb(t: float) -> torch.Tensor:
    from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
    cpu = torch.device('cpu')
    t_tensor = torch.tensor([t], dtype=torch.float32, device=cpu)
    emb = create_sinusoidal_pos_embedding(t_tensor, TIMESTEP_EMBED_DIM, MIN_PERIOD, MAX_PERIOD, device=cpu)
    return emb.to(torch.bfloat16).expand(BATCH_SIZE, -1)


def _make_dummy_inputs(seed=0):
    torch.manual_seed(seed)
    B = BATCH_SIZE
    pixel_values = torch.randn(B, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE, dtype=torch.bfloat16)
    text_embs    = torch.randn(B, NUM_TEXT_TOKENS,  HIDDEN_SIZE, dtype=torch.bfloat16)
    state_embs   = torch.randn(B, NUM_STATE_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16)
    return pixel_values, text_embs, state_embs


def _load_compiled():
    vis_path = COMPILED_DIR / 'vision'      / 'vision_encoder.pt'
    vlm_path = COMPILED_DIR / 'vlm'         / 'vlm_backbone.pt'
    exp_path = COMPILED_DIR / 'action_expert' / 'action_expert.pt'
    for p in (vis_path, vlm_path, exp_path):
        if not p.exists():
            raise FileNotFoundError(f"Compiled NEFF missing: {p}\nRun compile_all.py first.")
    vis = torch.jit.load(str(vis_path))
    vlm = torch.jit.load(str(vlm_path))
    exp = torch.jit.load(str(exp_path))
    return vis, vlm, exp


def _timeit(fn, n_warmup, n_runs):
    """Run fn() n_warmup times (discarded), then n_runs times. Return list of ms."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _stats(times_ms):
    return {
        'mean':   statistics.mean(times_ms),
        'median': statistics.median(times_ms),
        'p95':    sorted(times_ms)[int(len(times_ms) * 0.95)],
        'min':    min(times_ms),
        'max':    max(times_ms),
        'std':    statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def _print_stats(label, s, unit='ms'):
    print(f"  {label:<30} mean={s['mean']:>8.1f}{unit}  median={s['median']:>8.1f}{unit}"
          f"  p95={s['p95']:>8.1f}{unit}  std={s['std']:>6.1f}{unit}")


# ---------------------------------------------------------------------------
# Neuron benchmark
# ---------------------------------------------------------------------------

def benchmark_neuron(num_denoising_steps: int, n_warmup: int, n_runs: int):
    print("\n" + "=" * 72)
    print("NEURON BENCHMARK (compiled NEFFs)")
    print("=" * 72)

    vis_model, vlm_model, exp_model = _load_compiled()
    print("  Compiled NEFFs loaded ✓")

    pixel_values, text_embs, state_embs = _make_dummy_inputs()
    B = BATCH_SIZE

    # Pre-run the pipeline once to JIT-warm the Neuron runtime
    print(f"  Warming up ({n_warmup} runs per subgraph) ...")

    # Step 1 inputs
    attn_mask  = torch.ones(B, PREFIX_SEQ_LEN, PREFIX_SEQ_LEN, dtype=torch.bool)
    pos_ids    = torch.arange(PREFIX_SEQ_LEN).unsqueeze(0).expand(B, -1).clone()
    cross_mask = torch.ones(B, 1, ACTION_CHUNK_SIZE, NUM_CONDITIONING_TOKENS, dtype=torch.int32)
    dt         = -1.0 / num_denoising_steps
    times_seq  = [1.0 - i / num_denoising_steps for i in range(num_denoising_steps)]
    t_embs     = [_sinusoidal_emb(t) for t in times_seq]

    # -- Vision encoder --
    def run_vision():
        return vis_model(pixel_values)

    vis_times = _timeit(run_vision, n_warmup, n_runs)
    vision_embs = run_vision()
    prefix = torch.cat([text_embs, vision_embs.to(torch.bfloat16), state_embs], dim=1)

    # -- VLM backbone --
    def run_vlm():
        return vlm_model(prefix, attn_mask, pos_ids)

    vlm_times = _timeit(run_vlm, n_warmup, n_runs)
    conditioning = vlm_model(prefix, attn_mask, pos_ids).to(torch.bfloat16)

    # -- Action expert (single step) --
    x_t = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, dtype=torch.bfloat16)
    def run_expert_step():
        return exp_model(x_t, conditioning, t_embs[0], cross_mask)

    expert_step_times = _timeit(run_expert_step, n_warmup, n_runs)

    # -- Full denoising loop --
    def run_full_loop():
        x = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, dtype=torch.bfloat16)
        for t_emb in t_embs:
            v = exp_model(x, conditioning, t_emb, cross_mask)
            x = x + dt * v.to(x.dtype)
        return x

    loop_times = _timeit(run_full_loop, n_warmup, n_runs)

    # -- End-to-end (vision + VLM + full loop) --
    def run_e2e():
        ve = vis_model(pixel_values)
        pf = torch.cat([text_embs, ve.to(torch.bfloat16), state_embs], dim=1)
        cond = vlm_model(pf, attn_mask, pos_ids).to(torch.bfloat16)
        x = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, dtype=torch.bfloat16)
        for t_emb in t_embs:
            v = exp_model(x, cond, t_emb, cross_mask)
            x = x + dt * v.to(x.dtype)
        return x

    e2e_times = _timeit(run_e2e, n_warmup, n_runs)

    print(f"\n  Results ({n_runs} runs, batch_size={B}, denoising_steps={num_denoising_steps}):\n")
    _print_stats("Vision encoder",        _stats(vis_times))
    _print_stats("VLM backbone",          _stats(vlm_times))
    _print_stats(f"Expert step (×1)",     _stats(expert_step_times))
    loop_stats = _stats(loop_times)
    _print_stats(f"Expert loop (×{num_denoising_steps})", loop_stats)
    e2e_stats = _stats(e2e_times)
    _print_stats("End-to-end total",      e2e_stats)

    hz = 1000.0 / e2e_stats['mean']
    print(f"\n  Throughput: {hz:.2f} inferences/sec  ({B / (e2e_stats['mean']/1000):.2f} samples/sec)")

    return {
        'vision_ms':      _stats(vis_times),
        'vlm_ms':         _stats(vlm_times),
        'expert_step_ms': _stats(expert_step_times),
        'expert_loop_ms': loop_stats,
        'e2e_ms':         e2e_stats,
        'throughput_hz':  hz,
    }


# ---------------------------------------------------------------------------
# CPU reference benchmark
# ---------------------------------------------------------------------------

def benchmark_cpu_reference(num_denoising_steps: int, n_warmup: int, n_runs: int):
    print("\n" + "=" * 72)
    print("CPU REFERENCE BENCHMARK (HuggingFace model, float32)")
    print("=" * 72)

    try:
        from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
    except ImportError:
        print("  lerobot not available — skipping CPU reference")
        return None

    if not CHECKPOINT.exists():
        print(f"  Checkpoint not found at {CHECKPOINT} — skipping CPU reference")
        return None

    print("  Loading HF model (this may take a moment) ...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, SmolVLAConfig
    try:
        cfg = SmolVLAConfig()
        policy = SmolVLAPolicy(cfg)
        policy.eval()
        sd = load_file(str(CHECKPOINT))
        policy.load_state_dict(sd, strict=False)
        print("  HF model loaded ✓")
    except Exception as e:
        print(f"  Could not load HF model: {e} — skipping CPU reference")
        return None

    # Benchmark just the VLM prefix pass as a proxy (full policy is complex to time cleanly)
    from config_constants import NUM_HIDDEN_LAYERS
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    vlm_ref = SmolVLMWithExpertModel(load_vlm_weights=False, num_vlm_layers=NUM_HIDDEN_LAYERS)
    vlm_ref._attn_implementation = "eager"
    vlm_ref.eval()

    B = BATCH_SIZE
    prefix = torch.randn(B, PREFIX_SEQ_LEN, HIDDEN_SIZE, dtype=torch.float32)
    pos_ids = torch.arange(PREFIX_SEQ_LEN).unsqueeze(0).expand(B, -1)
    attn_mask = torch.ones(B, PREFIX_SEQ_LEN, PREFIX_SEQ_LEN, dtype=torch.bool)

    def run_vlm_ref():
        with torch.no_grad():
            return vlm_ref(
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=None,
                inputs_embeds=[prefix, None],
                use_cache=True,
                fill_kv_cache=True,
            )

    print(f"  Warming up VLM reference ({n_warmup} runs) ...")
    vlm_ref_times = _timeit(run_vlm_ref, n_warmup, min(n_runs, 10))

    print(f"\n  Results ({min(n_runs, 10)} runs, batch_size={B}):\n")
    _print_stats("VLM prefix pass (CPU f32)", _stats(vlm_ref_times))

    return {'vlm_cpu_ms': _stats(vlm_ref_times)}


# ---------------------------------------------------------------------------
# Speedup summary
# ---------------------------------------------------------------------------

def print_speedup_summary(neuron_results, cpu_results):
    if cpu_results is None or neuron_results is None:
        return
    print("\n" + "=" * 72)
    print("SPEEDUP SUMMARY")
    print("=" * 72)
    cpu_vlm   = cpu_results['vlm_cpu_ms']['mean']
    neuron_vlm = neuron_results['vlm_ms']['mean']
    print(f"  VLM backbone: {cpu_vlm:.1f}ms (CPU f32) → {neuron_vlm:.1f}ms (Neuron bf16)"
          f"  = {cpu_vlm/neuron_vlm:.1f}× speedup")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SmolVLA Trainium port benchmark")
    parser.add_argument('--warmup',       type=int, default=20,  help='Warmup runs (default: 20)')
    parser.add_argument('--runs',         type=int, default=50,  help='Timed runs (default: 50)')
    parser.add_argument('--steps',        type=int, default=NUM_DENOISING_STEPS,
                        help=f'Denoising steps (default: {NUM_DENOISING_STEPS})')
    parser.add_argument('--no-reference', action='store_true',   help='Skip CPU reference benchmark')
    args = parser.parse_args()

    print(f"SmolVLA Trainium Port — Benchmark")
    print(f"  batch_size={BATCH_SIZE}, denoising_steps={args.steps}")
    print(f"  warmup={args.warmup}, timed_runs={args.runs}")

    neuron_results = benchmark_neuron(args.steps, args.warmup, args.runs)

    cpu_results = None
    if not args.no_reference:
        cpu_results = benchmark_cpu_reference(args.steps, args.warmup // 4, args.runs // 5)

    print_speedup_summary(neuron_results, cpu_results)

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
