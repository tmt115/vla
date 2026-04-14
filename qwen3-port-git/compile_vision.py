"""
Deliverable 3: Compile Qwen3-VL vision encoder NEFF for Trainium.

Produces compiled artifacts in ~/qwen-port/compiled/vision/:
  - config.json              (Qwen3VLInferenceConfig saved by compile())
  - model.pt                 (TorchScript / NEFF from torch.jit.save)
  - NeuronQwen3VLVisionModel/ subdirs with per-bucket .neff files

Because the HF Qwen/Qwen3-VL-7B-Instruct model is gated on the Hub, this
script creates a minimal *fake* checkpoint that contains only the weight
actually loaded during model construction:
    model.visual.pos_embed.weight  [2304, 1152]  (bfloat16)

Weight sharding is skipped (save_sharded_checkpoint=False by default), so
no other weights are needed at compile time.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd ~/qwen-port
    python compile_vision.py [--checkpoint PATH] [--output OUTPUT]

Options:
    --checkpoint PATH   Path to a real HF checkpoint (optional).  If omitted,
                        a minimal fake checkpoint is created in /tmp/.
    --output OUTPUT     Output directory for compiled artifacts.
                        Default: ~/qwen-port/compiled/vision
    --dry-run           Trace only (no NEFF written to disk).
"""

import argparse
import os
import sys

NXDI_ROOT = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages"
if NXDI_ROOT not in sys.path:
    sys.path.insert(0, NXDI_ROOT)

import torch


# ── Constants from config_constants.py ────────────────────────────────────────
from config_constants import (
    # Vision encoder
    VIS_DEPTH,
    VIS_HIDDEN_SIZE,
    VIS_INTERMEDIATE_SIZE,
    VIS_NUM_HEADS,
    VIS_IN_CHANNELS,
    VIS_PATCH_SIZE,
    VIS_TEMPORAL_PATCH_SIZE,
    VIS_SPATIAL_MERGE_SIZE,
    VIS_OUT_HIDDEN_SIZE,
    VIS_NUM_POSITION_EMBEDDINGS,
    VIS_HIDDEN_ACT,
    VIS_DEEPSTACK_VISUAL_INDEXES,
    # Text decoder (needed for InferenceConfig construction)
    TEXT_HIDDEN_SIZE,
    TEXT_NUM_ATTENTION_HEADS,
    TEXT_NUM_KEY_VALUE_HEADS,
    TEXT_NUM_HIDDEN_LAYERS,
    TEXT_INTERMEDIATE_SIZE,
    TEXT_HEAD_DIM,
    TEXT_VOCAB_SIZE,
    TEXT_RMS_NORM_EPS,
    TEXT_ROPE_THETA,
    TEXT_MAX_POSITION_EMBEDDINGS,
    TEXT_ROPE_SCALING,
    # Inference / Neuron
    TP_DEGREE,
    MAX_BATCH_SIZE,
    TEXT_SEQ_LEN,
    TEXT_MAX_CONTEXT_LENGTH,
    TORCH_DTYPE,
    VISION_BUCKETS,
    VISION_SEQ_LEN,
)


def make_fake_checkpoint(path: str) -> None:
    """
    Write a minimal safetensors checkpoint containing only the one weight
    that NeuronQwen3VLVisionModelWrapper.__init__ loads:
        model.visual.pos_embed.weight  [VIS_NUM_POSITION_EMBEDDINGS, VIS_HIDDEN_SIZE]
    """
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    weight = torch.zeros(
        VIS_NUM_POSITION_EMBEDDINGS, VIS_HIDDEN_SIZE, dtype=torch.bfloat16
    )
    save_file({"model.visual.pos_embed.weight": weight},
              os.path.join(path, "model.safetensors"))
    print(f"[compile_vision] Fake checkpoint written to {path}")


def build_config(checkpoint_path: str):
    """
    Build a Qwen3VLInferenceConfig for the 7B-Instruct production configuration.
    tp_degree=8 matches a trn1.32xlarge with 32 NeuronCores (4 per device × 8).
    """
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )
    from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLInferenceConfig,
        Qwen3VLNeuronConfig,
    )

    # ── HF vision config ──────────────────────────────────────────────────────
    hf_vc = Qwen3VLVisionConfig(
        depth=VIS_DEPTH,
        hidden_size=VIS_HIDDEN_SIZE,
        intermediate_size=VIS_INTERMEDIATE_SIZE,
        num_heads=VIS_NUM_HEADS,
        in_channels=VIS_IN_CHANNELS,
        patch_size=VIS_PATCH_SIZE,
        temporal_patch_size=VIS_TEMPORAL_PATCH_SIZE,
        spatial_merge_size=VIS_SPATIAL_MERGE_SIZE,
        out_hidden_size=VIS_OUT_HIDDEN_SIZE,
        num_position_embeddings=VIS_NUM_POSITION_EMBEDDINGS,
        hidden_act=VIS_HIDDEN_ACT,
    )
    hf_vc.deepstack_visual_indexes = VIS_DEEPSTACK_VISUAL_INDEXES

    # ── HF text config ─────────────────────────────────────────────────────────
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

    # ── Neuron configs ─────────────────────────────────────────────────────────
    vision_nc = Qwen3VLNeuronConfig(
        tp_degree=TP_DEGREE,
        torch_dtype=TORCH_DTYPE,
        batch_size=MAX_BATCH_SIZE,
        seq_len=VISION_SEQ_LEN,
        n_active_tokens=VISION_SEQ_LEN,
        on_device_sampling_config=None,
        buckets=VISION_BUCKETS,
        enable_bucketing=True,
    )

    text_nc = Qwen3VLNeuronConfig(
        tp_degree=TP_DEGREE,
        torch_dtype=TORCH_DTYPE,
        batch_size=MAX_BATCH_SIZE,
        seq_len=TEXT_SEQ_LEN,
        n_active_tokens=TEXT_SEQ_LEN,
        max_context_length=TEXT_MAX_CONTEXT_LENGTH,
        on_device_sampling_config=None,
        buckets=[TEXT_SEQ_LEN],
    )

    cfg = Qwen3VLInferenceConfig(
        text_neuron_config=text_nc,
        vision_neuron_config=vision_nc,
        text_config=vars(hf_tc),
        vision_config=vars(hf_vc),
        _name_or_path=checkpoint_path,
    )
    return cfg


def compile_vision(checkpoint_path: str, output_dir: str, dry_run: bool = False) -> None:
    from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
        NeuronQwen3VLForImageEncoding,
    )

    print(f"[compile_vision] checkpoint : {checkpoint_path}")
    print(f"[compile_vision] output dir : {output_dir}")
    print(f"[compile_vision] dry_run    : {dry_run}")
    print(f"[compile_vision] tp_degree  : {TP_DEGREE}")
    print(f"[compile_vision] buckets    : {VISION_BUCKETS}")

    cfg = build_config(checkpoint_path)
    print("[compile_vision] Config built.")

    app = NeuronQwen3VLForImageEncoding(checkpoint_path, cfg)
    print("[compile_vision] NeuronQwen3VLForImageEncoding instantiated.")

    os.makedirs(output_dir, exist_ok=True)
    app.compile(output_dir, dry_run=dry_run)

    if dry_run:
        print("[compile_vision] Dry-run complete (trace only, no NEFF saved).")
    else:
        print(f"[compile_vision] Compilation complete. Artifacts in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a real HF checkpoint. If omitted, a minimal fake checkpoint is used.",
    )
    parser.add_argument(
        "--output", default=os.path.expanduser("~/qwen-port/compiled/vision"),
        help="Output directory for compiled artifacts (default: ~/qwen-port/compiled/vision).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Trace only; do not write NEFF to disk.",
    )
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_path = os.path.expanduser(args.checkpoint)
        if not os.path.isdir(checkpoint_path):
            parser.error(f"Checkpoint path does not exist or is not a directory: {checkpoint_path}")
    else:
        checkpoint_path = "/tmp/fake_qwen3vl_ckpt"
        make_fake_checkpoint(checkpoint_path)

    compile_vision(
        checkpoint_path=checkpoint_path,
        output_dir=os.path.expanduser(args.output),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
