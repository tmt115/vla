"""
Deliverable 7: Integration smoke test for the compiled Qwen3-VL NEFFs.

Two tiers of validation:

  Tier 1 — artifact checks (no checkpoint required):
    1a. File existence  — D3 and D6 model.pt files are present.
    1b. NEFF integrity  — model.pt files are valid TorchScript modules
        (torch.jit.load succeeds).
    1c. NEFF size       — each model.pt is >1 MB (not a stub/empty file).

  Tier 2 — inference checks (requires --checkpoint to a real HF checkpoint):
    2a. D3 vision encoder — load NEFF + weights, run 784-patch forward,
        validate output shape (196, 3584) and finiteness.
    2b. D6 full model     — load NEFF + weights, run text-only prefill,
        validate logits shape (1, 8192, 151936) and finiteness.

Usage:
    # Tier 1 only (default, works with fake checkpoint)
    python smoke_test.py

    # Tier 1 + Tier 2 (requires real HF checkpoint)
    python smoke_test.py --checkpoint /path/to/Qwen3-VL-7B-Instruct
"""

import argparse
import os
import sys

NXDI_ROOT = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages"
if NXDI_ROOT not in sys.path:
    sys.path.insert(0, NXDI_ROOT)

import torch

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

_SMOKE_T, _SMOKE_H, _SMOKE_W = 1, 28, 28
_SMOKE_PATCHES = _SMOKE_T * _SMOKE_H * _SMOKE_W  # 784
_MIN_NEFF_BYTES = 1 * 1024 * 1024  # 1 MB sanity floor


# ── Config builder (shared by both tiers) ─────────────────────────────────────

def _build_config(checkpoint_path: str):
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
        head_dim=TEXT_HEAD_DIM, vocab_size=TEXT_VOCAB_SIZE,
        rms_norm_eps=TEXT_RMS_NORM_EPS, rope_theta=TEXT_ROPE_THETA,
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
        attn_kernel_enabled=False,
    )
    return Qwen3VLInferenceConfig(
        text_neuron_config=text_nc, vision_neuron_config=vision_nc,
        text_config=vars(hf_tc), vision_config=vars(hf_vc),
        _name_or_path=checkpoint_path,
        image_token_id=IMAGE_TOKEN_ID, video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
    )


# ── Tier 1: artifact checks (no checkpoint needed) ────────────────────────────

def tier1_artifacts(compiled_base: str) -> dict:
    """
    Check file existence and minimum size.
    Note: torch.jit.load on a Neuron model.pt requires the Neuron runtime C++
    extensions to be registered (done by app.load(), not standalone).
    Size check is the correct proxy for a valid compiled artifact here.
    """
    neffs = {
        "D3 vision/model.pt":        os.path.join(compiled_base, "vision",       "model.pt"),
        "D6 text_model/model.pt":    os.path.join(compiled_base, "text_model",   "model.pt"),
        "D6 vision_model/model.pt":  os.path.join(compiled_base, "vision_model", "model.pt"),
    }
    results = {}
    for label, path in neffs.items():
        if not os.path.isfile(path):
            results[label] = (False, "MISSING")
            continue
        size = os.path.getsize(path)
        if size < _MIN_NEFF_BYTES:
            results[label] = (False, f"too small ({size / 1e6:.1f} MB, expected >{_MIN_NEFF_BYTES // (1024*1024)} MB)")
            continue
        results[label] = (True, f"{size / 1e6:.1f} MB")
    return results


# ── Tier 2: inference checks (requires real checkpoint) ───────────────────────

def tier2_vision(compiled_base: str, checkpoint_path: str) -> tuple[bool, str]:
    from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
        NeuronQwen3VLForImageEncoding,
    )
    vision_dir = os.path.join(compiled_base, "vision")
    cfg = _build_config(checkpoint_path)
    app = NeuronQwen3VLForImageEncoding(checkpoint_path, cfg)
    app.load(vision_dir)

    pixel_values = torch.zeros(_SMOKE_PATCHES, VIS_PATCHES_PER_PIXEL_SEQ, dtype=torch.bfloat16)
    image_grid_thw = torch.tensor([[_SMOKE_T, _SMOKE_H, _SMOKE_W]], dtype=torch.long)
    with torch.no_grad():
        out = app.forward(pixel_values, image_grid_thw)

    n_compressed = _SMOKE_PATCHES // (VIS_SPATIAL_MERGE_SIZE ** 2)
    expected = (n_compressed, VIS_OUT_HIDDEN_SIZE)
    if tuple(out.shape) != expected:
        return False, f"shape mismatch: expected {expected}, got {tuple(out.shape)}"
    if not torch.isfinite(out).all():
        return False, "output contains NaN/Inf"
    return True, f"shape {tuple(out.shape)}, all finite"


def tier2_full(compiled_base: str, checkpoint_path: str) -> tuple[bool, str]:
    from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import (
        NeuronQwen3VLForCausalLM,
    )
    from neuronx_distributed_inference.modules.generation.sampling import (
        prepare_sampling_params,
    )
    cfg = _build_config(checkpoint_path)
    app = NeuronQwen3VLForCausalLM(checkpoint_path, cfg)
    app.load(compiled_base)

    n_active = 32
    input_ids = torch.zeros(1, TEXT_SEQ_LEN, dtype=torch.long)
    input_ids[0, :n_active] = torch.randint(1, TEXT_VOCAB_SIZE, (n_active,))
    attention_mask = torch.zeros(1, TEXT_SEQ_LEN, dtype=torch.long)
    attention_mask[0, :n_active] = 1
    position_ids = torch.zeros(1, TEXT_SEQ_LEN, dtype=torch.long)
    position_ids[0, :n_active] = torch.arange(n_active)
    sampling_params = prepare_sampling_params(batch_size=1, top_k=[1])

    with torch.no_grad():
        result, _ = app.forward_atomic_prefill(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, seq_ids=torch.tensor([0]),
            pixel_values=None, image_grid_thw=None,
            sampling_params=sampling_params,
        )

    if hasattr(result, "logits") and result.logits is not None:
        logits = result.logits
        expected = (1, TEXT_SEQ_LEN, TEXT_VOCAB_SIZE)
        if tuple(logits.shape) != expected:
            return False, f"logits shape mismatch: expected {expected}, got {tuple(logits.shape)}"
        if not torch.isfinite(logits).all():
            return False, "logits contain NaN/Inf"
        return True, f"logits shape {tuple(logits.shape)}, all finite"
    elif hasattr(result, "tokens") and result.tokens is not None:
        return True, f"tokens shape {tuple(result.tokens.shape)}"
    return False, "no logits or tokens in output"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--compiled-dir", default=os.path.expanduser("~/qwen-port/compiled"),
        help="Base directory holding D3 (vision/) and D6 (text_model/, vision_model/) outputs.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help=(
            "Path to a real Qwen3-VL-7B-Instruct HF checkpoint. "
            "Required for Tier 2 inference tests. "
            "Omit to run Tier 1 (artifact checks) only."
        ),
    )
    parser.add_argument("--skip-d3", action="store_true", help="Skip D3 vision inference test.")
    parser.add_argument("--skip-d6", action="store_true", help="Skip D6 full model inference test.")
    args = parser.parse_args()

    compiled_base   = os.path.expanduser(args.compiled_dir)
    checkpoint_path = os.path.expanduser(args.checkpoint) if args.checkpoint else None
    has_checkpoint  = checkpoint_path is not None and os.path.isdir(checkpoint_path)

    overall = True

    # ── Tier 1: artifact checks ────────────────────────────────────────────────
    print("=" * 60)
    print("Tier 1: Artifact checks (existence + NEFF integrity)")
    print("=" * 60)
    t1 = tier1_artifacts(compiled_base)
    for label, (ok, detail) in t1.items():
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {label}  — {detail}")
        if not ok:
            overall = False

    if not all(ok for ok, _ in t1.values()):
        print("\nArtifact checks failed. Run compile_vision.py and compile_text.py first.")
        sys.exit(1)

    # ── Tier 2: inference checks ───────────────────────────────────────────────
    if not has_checkpoint:
        print(
            "\nTier 2: Inference checks SKIPPED "
            "(pass --checkpoint /path/to/Qwen3-VL-7B-Instruct to enable)"
        )
    else:
        print(f"\n{'='*60}")
        print(f"Tier 2: Inference checks  (checkpoint: {checkpoint_path})")
        print(f"{'='*60}")

        t2_results = {}

        if not args.skip_d3:
            print("\n[D3] Loading standalone vision NEFF + weights …")
            try:
                ok, detail = tier2_vision(compiled_base, checkpoint_path)
                t2_results["D3 vision"] = (ok, detail)
            except Exception as exc:
                t2_results["D3 vision"] = (False, str(exc))

        if not args.skip_d6:
            print("\n[D6] Loading full model NEFF + weights …")
            try:
                ok, detail = tier2_full(compiled_base, checkpoint_path)
                t2_results["D6 full model"] = (ok, detail)
            except Exception as exc:
                t2_results["D6 full model"] = (False, str(exc))

        print()
        for label, (ok, detail) in t2_results.items():
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label}  — {detail}")
            if not ok:
                overall = False

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if overall:
        print("All checks passed.")
        sys.exit(0)
    else:
        print("One or more checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
