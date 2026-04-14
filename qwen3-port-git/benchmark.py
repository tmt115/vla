"""
Benchmark for compiled Qwen3-VL NEFFs on Trainium.

Measures latency of:
  - Vision encoding          (standalone vision encoder NEFF, D3)
  - Context encoding/prefill (vision encode + CTE, D6)
  - Single token generation  (TKG step, D6)
  - Full inference           (prefill + N decode steps, D6)

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd ~/qwen-port
    python benchmark.py [options]

Options:
    --compiled-dir PATH   Base compiled dir (default: ~/qwen-port/compiled)
    --checkpoint PATH     Checkpoint used at compile time (default: /tmp/fake_qwen3vl_ckpt)
    --decode-steps N      Token generation steps per inference run (default: 20)
    --warmup N            Warmup iterations (default: 10)
    --runs N              Timed iterations (default: 100)
    --vision-bucket B     Vision bucket to benchmark: 784, 3136, or 12544 (default: 784)
    --skip-vision         Skip standalone vision benchmark (D3)
    --skip-tkg            Skip token generation step benchmark
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

NXDI_ROOT = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages"
if NXDI_ROOT not in sys.path:
    sys.path.insert(0, NXDI_ROOT)

from config_constants import (
    VIS_DEPTH, VIS_HIDDEN_SIZE, VIS_INTERMEDIATE_SIZE, VIS_NUM_HEADS,
    VIS_IN_CHANNELS, VIS_PATCH_SIZE, VIS_TEMPORAL_PATCH_SIZE,
    VIS_SPATIAL_MERGE_SIZE, VIS_OUT_HIDDEN_SIZE, VIS_NUM_POSITION_EMBEDDINGS,
    VIS_HIDDEN_ACT, VIS_DEEPSTACK_VISUAL_INDEXES, VIS_PATCHES_PER_PIXEL_SEQ,
    TEXT_HIDDEN_SIZE, TEXT_NUM_ATTENTION_HEADS, TEXT_NUM_KEY_VALUE_HEADS,
    TEXT_NUM_HIDDEN_LAYERS, TEXT_INTERMEDIATE_SIZE, TEXT_HEAD_DIM,
    TEXT_VOCAB_SIZE, TEXT_RMS_NORM_EPS, TEXT_ROPE_THETA,
    TEXT_MAX_POSITION_EMBEDDINGS, TEXT_ROPE_SCALING,
    TP_DEGREE, MAX_BATCH_SIZE, TEXT_SEQ_LEN, TEXT_MAX_CONTEXT_LENGTH,
    TORCH_DTYPE, VISION_BUCKETS, VISION_SEQ_LEN,
    IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, VISION_START_TOKEN_ID, VISION_END_TOKEN_ID,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _percentile(arr, p):
    return float(np.percentile(arr, p))

def _stats(times_ms):
    return {
        "mean": float(np.mean(times_ms)),
        "p50":  _percentile(times_ms, 50),
        "p99":  _percentile(times_ms, 99),
        "min":  float(np.min(times_ms)),
        "max":  float(np.max(times_ms)),
    }

def _print_stats(label, s):
    print(f"  {label}")
    print(f"    Mean  : {s['mean']:.1f} ms")
    print(f"    P50   : {s['p50']:.1f} ms")
    print(f"    P99   : {s['p99']:.1f} ms")
    print(f"    Min   : {s['min']:.1f} ms")
    print(f"    Max   : {s['max']:.1f} ms")

def _time_fn(fn, warmup, runs):
    """Run fn() warmup+runs times; return list of timed latencies in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _build_config(checkpoint_path):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig, Qwen3VLVisionConfig,
    )
    from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLInferenceConfig, Qwen3VLNeuronConfig,
    )

    hf_vc = Qwen3VLVisionConfig(
        depth=VIS_DEPTH, hidden_size=VIS_HIDDEN_SIZE,
        intermediate_size=VIS_INTERMEDIATE_SIZE, num_heads=VIS_NUM_HEADS,
        in_channels=VIS_IN_CHANNELS, patch_size=VIS_PATCH_SIZE,
        temporal_patch_size=VIS_TEMPORAL_PATCH_SIZE,
        spatial_merge_size=VIS_SPATIAL_MERGE_SIZE,
        out_hidden_size=VIS_OUT_HIDDEN_SIZE,
        num_position_embeddings=VIS_NUM_POSITION_EMBEDDINGS,
        hidden_act=VIS_HIDDEN_ACT,
    )
    hf_vc.deepstack_visual_indexes = VIS_DEEPSTACK_VISUAL_INDEXES

    hf_tc = Qwen3VLTextConfig(
        hidden_size=TEXT_HIDDEN_SIZE,
        num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
        num_key_value_heads=TEXT_NUM_KEY_VALUE_HEADS,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
        num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
        head_dim=TEXT_HEAD_DIM,
        vocab_size=TEXT_VOCAB_SIZE,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        rope_theta=TEXT_ROPE_THETA,
        rope_scaling=TEXT_ROPE_SCALING,
        max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
    )
    vision_nc = Qwen3VLNeuronConfig(
        tp_degree=TP_DEGREE, torch_dtype=TORCH_DTYPE, batch_size=MAX_BATCH_SIZE,
        seq_len=VISION_SEQ_LEN, n_active_tokens=VISION_SEQ_LEN,
        on_device_sampling_config=None, buckets=VISION_BUCKETS,
        enable_bucketing=True,
    )
    text_nc = Qwen3VLNeuronConfig(
        tp_degree=TP_DEGREE, torch_dtype=TORCH_DTYPE, batch_size=MAX_BATCH_SIZE,
        seq_len=TEXT_SEQ_LEN, n_active_tokens=TEXT_SEQ_LEN,
        max_context_length=TEXT_MAX_CONTEXT_LENGTH,
        on_device_sampling_config=None, buckets=[TEXT_SEQ_LEN],
        attn_kernel_enabled=False,  # NeuronX 2.24: NKI CTE kernel bugs at q_len>=4096
    )
    return Qwen3VLInferenceConfig(
        text_neuron_config=text_nc, vision_neuron_config=vision_nc,
        text_config=vars(hf_tc), vision_config=vars(hf_vc),
        _name_or_path=checkpoint_path,
        image_token_id=IMAGE_TOKEN_ID, video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
    )


def _make_vision_inputs(vision_bucket: int):
    """
    Create dummy pixel_values and image_grid_thw for a given vision bucket.
    vision_bucket must be one of VISION_BUCKETS (784, 3136, 12544).
    For simplicity, always use a single frame (T=1).
    """
    assert vision_bucket in VISION_BUCKETS, f"Expected one of {VISION_BUCKETS}"
    # Compute H×W = vision_bucket (all single-frame)
    hw = vision_bucket   # T=1 so H*W = vision_bucket
    import math
    h = w = int(math.isqrt(hw))
    assert h * w == hw, f"vision_bucket {hw} is not a perfect square"

    pixel_values = torch.zeros(
        vision_bucket, VIS_PATCHES_PER_PIXEL_SEQ, dtype=torch.bfloat16
    )
    image_grid_thw = torch.tensor([[1, h, w]], dtype=torch.long)
    # Compressed token count after spatial merge
    n_image_tokens = vision_bucket // (VIS_SPATIAL_MERGE_SIZE ** 2)
    return pixel_values, image_grid_thw, n_image_tokens


def _make_prefill_inputs(n_image_tokens: int):
    """
    Build a [1, TEXT_SEQ_LEN] text sequence containing:
      - a few leading text tokens (system prompt stand-in)
      - n_image_tokens image token ids (IMAGE_TOKEN_ID)
      - trailing text tokens (user query stand-in)
      - zero-padded to TEXT_SEQ_LEN

    Returns input_ids, attention_mask, position_ids, seq_ids, n_active
    """
    n_pre  = 10   # tokens before the image
    n_post = 10   # tokens after the image
    n_active = n_pre + n_image_tokens + n_post

    ids = torch.zeros(1, TEXT_SEQ_LEN, dtype=torch.long)
    ids[0, :n_pre]                          = 1   # BOS-like
    ids[0, n_pre:n_pre+n_image_tokens]      = IMAGE_TOKEN_ID
    ids[0, n_pre+n_image_tokens:n_active]   = 2   # simple text tokens

    mask = torch.zeros(1, TEXT_SEQ_LEN, dtype=torch.long)
    mask[0, :n_active] = 1

    pos = torch.zeros(1, TEXT_SEQ_LEN, dtype=torch.long)
    pos[0, :n_active] = torch.arange(n_active)

    seq_ids = torch.tensor([0], dtype=torch.long)
    return ids, mask, pos, seq_ids, n_active


def _make_tkg_inputs(n_active: int, next_token_id: int = 3):
    """Inputs for a single token-generation step after a prefill of n_active tokens."""
    input_ids    = torch.tensor([[next_token_id]], dtype=torch.long)
    attention_mask = torch.zeros(1, TEXT_SEQ_LEN, dtype=torch.long)
    attention_mask[0, :n_active + 1] = 1   # include the new token
    position_ids = torch.tensor([[n_active]], dtype=torch.long)  # min > 0 → TKG
    seq_ids      = torch.tensor([0], dtype=torch.long)
    return input_ids, attention_mask, position_ids, seq_ids


# ── Benchmark D3: standalone vision encoder ────────────────────────────────────

def bench_vision(compiled_base, checkpoint_path, vision_bucket, warmup, runs):
    from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
        NeuronQwen3VLForImageEncoding,
    )

    vision_dir = os.path.join(compiled_base, "vision")
    if not os.path.isfile(os.path.join(vision_dir, "model.pt")):
        print(f"[vision] SKIP — {vision_dir}/model.pt not found (run compile_vision.py first)")
        return None

    print(f"\n[vision] Loading standalone vision NEFF from {vision_dir} …")
    cfg = _build_config(checkpoint_path)
    app = NeuronQwen3VLForImageEncoding(checkpoint_path, cfg)
    app.load(vision_dir)

    pixel_values, image_grid_thw, n_img_tok = _make_vision_inputs(vision_bucket)
    print(f"[vision] vision_bucket={vision_bucket} patches → {n_img_tok} compressed tokens")

    def fn():
        with torch.no_grad():
            app.forward(pixel_values, image_grid_thw)

    print(f"[vision] Benchmarking: {warmup} warmup + {runs} timed runs …")
    times = _time_fn(fn, warmup, runs)

    out = app.forward(pixel_values, image_grid_thw)
    print(f"\n{'='*52}")
    print(f"  Vision encoder  (bucket={vision_bucket})")
    print(f"  Output shape : {tuple(out.shape)}  dtype={out.dtype}")
    _print_stats("Latency:", _stats(times))
    print(f"{'='*52}")
    return _stats(times)


# ── Benchmark D6: prefill + TKG ────────────────────────────────────────────────

def bench_full(compiled_base, checkpoint_path, vision_bucket,
               decode_steps, warmup, runs, skip_tkg):
    from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import (
        NeuronQwen3VLForCausalLM,
    )
    from neuronx_distributed_inference.modules.generation.sampling import (
        prepare_sampling_params,
    )

    if not os.path.isfile(os.path.join(compiled_base, "text_model", "model.pt")):
        print(f"[full] SKIP — {compiled_base}/text_model/model.pt not found (run compile_text.py first)")
        return

    print(f"\n[full] Loading full model NEFFs from {compiled_base} …")
    cfg = _build_config(checkpoint_path)
    app = NeuronQwen3VLForCausalLM(checkpoint_path, cfg)
    app.load(compiled_base)

    pixel_values, image_grid_thw, n_img_tok = _make_vision_inputs(vision_bucket)
    input_ids, attn_mask, pos_ids, seq_ids, n_active = _make_prefill_inputs(n_img_tok)
    sampling_params = prepare_sampling_params(batch_size=1, top_k=[1])

    print(f"[full] vision_bucket={vision_bucket}, n_active_tokens={n_active}, "
          f"decode_steps={decode_steps}")

    # ── Phase A: time vision encode alone ─────────────────────────────────────
    def vision_fn():
        with torch.no_grad():
            app.vision_encoder_model(pixel_values, image_grid_thw)

    print(f"\n[full] Phase A — vision encode: {warmup} warmup + {runs} timed …")
    ve_times = _time_fn(vision_fn, warmup, runs)

    # ── Phase B: time full prefill (vision encode + CTE) ─────────────────────
    def prefill_fn():
        app.reset()
        with torch.no_grad():
            app.forward_atomic_prefill(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                seq_ids=seq_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                sampling_params=sampling_params,
            )

    print(f"[full] Phase B — full prefill: {warmup} warmup + {runs} timed …")
    prefill_times = _time_fn(prefill_fn, warmup, runs)

    # ── Phase C: time single TKG step ─────────────────────────────────────────
    tkg_times = None
    if not skip_tkg:
        # Do one fresh prefill to populate KV cache
        app.reset()
        with torch.no_grad():
            app.forward_atomic_prefill(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                seq_ids=seq_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                sampling_params=sampling_params,
            )

        tkg_ids, tkg_attn, tkg_pos, tkg_seq = _make_tkg_inputs(n_active)

        def tkg_fn():
            # position_ids.min() > 0 → routes to token_generation_model
            with torch.no_grad():
                app.forward(
                    input_ids=tkg_ids,
                    attention_mask=tkg_attn,
                    position_ids=tkg_pos,
                    seq_ids=tkg_seq,
                    sampling_params=sampling_params,
                )

        print(f"[full] Phase C — single TKG step: {warmup} warmup + {runs} timed …")
        tkg_times = _time_fn(tkg_fn, warmup, runs)

    # ── Results ───────────────────────────────────────────────────────────────
    ve_s       = _stats(ve_times)
    prefill_s  = _stats(prefill_times)
    cte_p50    = prefill_s["p50"] - ve_s["p50"]   # CTE-only estimate
    tkg_s      = _stats(tkg_times) if tkg_times is not None else None

    # Run one complete inference for result shape
    app.reset()
    with torch.no_grad():
        out, _ = app.forward_atomic_prefill(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            seq_ids=seq_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            sampling_params=sampling_params,
        )
    result_shape = tuple(out.logits.shape) if out.logits is not None else "n/a"
    result_dtype = out.logits.dtype if out.logits is not None else "n/a"

    if tkg_s is not None:
        est_total = prefill_s["p50"] + decode_steps * tkg_s["p50"]
    else:
        est_total = prefill_s["p50"]

    print(f"\n{'='*52}")
    print(f"  Full model  (vision_bucket={vision_bucket}, decode_steps={decode_steps})")
    print(f"  Prefill result shape : {result_shape}  dtype={result_dtype}")
    print()
    print(f"  Benchmarking: {warmup} warmup + {runs} timed runs")
    print()
    _print_stats("Vision encode (VE):", ve_s)
    print()
    _print_stats(f"Full prefill  (VE + CTE):", prefill_s)
    if tkg_s:
        print()
        _print_stats("Single TKG step:", tkg_s)
    print()
    print(f"  Per-component breakdown (p50):")
    print(f"    Vision encode         : {ve_s['p50']:.1f} ms")
    print(f"    Context encoding (CTE): {cte_p50:.1f} ms  (prefill - VE)")
    if tkg_s:
        print(f"    Single TKG step       : {tkg_s['p50']:.1f} ms")
        print(f"    Estimated total ({decode_steps:2d} steps): {est_total:.1f} ms")
    print(f"{'='*52}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--compiled-dir", default=os.path.expanduser("~/qwen-port/compiled"))
    parser.add_argument("--checkpoint",   default="/tmp/fake_qwen3vl_ckpt")
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--warmup",       type=int, default=10)
    parser.add_argument("--runs",         type=int, default=100)
    parser.add_argument("--vision-bucket",type=int, default=784,
                        choices=VISION_BUCKETS)
    parser.add_argument("--skip-vision",  action="store_true",
                        help="Skip standalone D3 vision benchmark")
    parser.add_argument("--skip-tkg",     action="store_true",
                        help="Skip TKG step benchmark")
    args = parser.parse_args()

    compiled_base   = os.path.expanduser(args.compiled_dir)
    checkpoint_path = os.path.expanduser(args.checkpoint)

    print(f"Compiled artifacts : {compiled_base}")
    print(f"Checkpoint         : {checkpoint_path}")
    print(f"Vision bucket      : {args.vision_bucket} patches")
    print(f"Decode steps       : {args.decode_steps}")
    print(f"Warmup / runs      : {args.warmup} / {args.runs}")

    if not args.skip_vision:
        bench_vision(
            compiled_base, checkpoint_path,
            args.vision_bucket, args.warmup, args.runs,
        )

    bench_full(
        compiled_base, checkpoint_path,
        args.vision_bucket, args.decode_steps,
        args.warmup, args.runs, args.skip_tkg,
    )


if __name__ == "__main__":
    main()
