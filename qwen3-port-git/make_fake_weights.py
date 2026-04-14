"""
Generate a complete fake Qwen3-VL-7B-Instruct checkpoint with random bfloat16 weights.

All 662 parameter tensors are created with the correct shapes (matching the
compiled NEFFs) but filled with random values.  Latency benchmarks depend only
on the NEFF graph structure, not the weight values, so this gives accurate
hardware performance numbers.

The checkpoint is saved in HF safetensors format — the same format that
NeuronQwen3VLForCausalLM.get_state_dict() / convert_hf_to_neuron_state_dict()
expect to load from.

Output: /tmp/fake_qwen3vl_full_ckpt/model.safetensors  (~14 GB bfloat16)

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd ~/qwen-port
    python make_fake_weights.py [--output PATH]
"""

import argparse
import os
import sys
import time

NXDI_ROOT = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages"
if NXDI_ROOT not in sys.path:
    sys.path.insert(0, NXDI_ROOT)

import torch
from safetensors.torch import save_file


def build_fake_checkpoint(output_dir: str) -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig, Qwen3VLVisionConfig, Qwen3VLTextConfig,
    )
    from config_constants import (
        VIS_DEPTH, VIS_HIDDEN_SIZE, VIS_INTERMEDIATE_SIZE, VIS_NUM_HEADS,
        VIS_IN_CHANNELS, VIS_PATCH_SIZE, VIS_TEMPORAL_PATCH_SIZE,
        VIS_SPATIAL_MERGE_SIZE, VIS_OUT_HIDDEN_SIZE, VIS_NUM_POSITION_EMBEDDINGS,
        VIS_HIDDEN_ACT,
        TEXT_HIDDEN_SIZE, TEXT_NUM_ATTENTION_HEADS, TEXT_NUM_KEY_VALUE_HEADS,
        TEXT_NUM_HIDDEN_LAYERS, TEXT_INTERMEDIATE_SIZE, TEXT_HEAD_DIM,
        TEXT_VOCAB_SIZE, TEXT_RMS_NORM_EPS, TEXT_ROPE_THETA,
        TEXT_MAX_POSITION_EMBEDDINGS, TEXT_ROPE_SCALING,
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
    cfg = Qwen3VLConfig(
        text_config=hf_tc.to_dict(),
        vision_config=hf_vc.to_dict(),
    )

    # ── Step 1: get shapes via meta device (no RAM allocated) ─────────────────
    print("Getting parameter shapes via meta device ...")
    with torch.device("meta"):
        meta_model = Qwen3VLForConditionalGeneration(cfg)
    shapes = {k: tuple(v.shape) for k, v in meta_model.state_dict().items()}
    print(f"  {len(shapes)} parameters")
    del meta_model

    # ── Step 2: allocate random bfloat16 tensors ──────────────────────────────
    total_bytes = sum(
        torch.tensor(0, dtype=torch.bfloat16).element_size() * torch.Size(s).numel()
        for s in shapes.values()
    )
    print(f"Allocating {total_bytes / 1e9:.1f} GB of random bfloat16 weights ...")

    state_dict = {}
    t0 = time.monotonic()
    for i, (k, shape) in enumerate(shapes.items()):
        state_dict[k] = torch.empty(shape, dtype=torch.bfloat16).uniform_(-0.02, 0.02)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(shapes)} tensors allocated ...")
    print(f"  Done in {time.monotonic()-t0:.1f}s")

    # ── Step 3: save ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.safetensors")
    print(f"Saving to {out_path} ...")
    t0 = time.monotonic()
    save_file(state_dict, out_path)
    print(f"  Saved in {time.monotonic()-t0:.1f}s  ({os.path.getsize(out_path)/1e9:.1f} GB)")

    # Also save a minimal config.json so HF tools recognise the directory
    cfg.save_pretrained(output_dir)
    print(f"Fake checkpoint ready: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="/tmp/fake_qwen3vl_full_ckpt",
                        help="Output directory (default: /tmp/fake_qwen3vl_full_ckpt)")
    args = parser.parse_args()
    build_fake_checkpoint(os.path.expanduser(args.output))


if __name__ == "__main__":
    main()
