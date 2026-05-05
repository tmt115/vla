"""
validate_neffs.py — NEFF correctness validation against CPU reference.

Runs after compile_all.py. Validates each NEFF against HF reference outputs.
Thresholds per SKILL.md:
  - Vision encoder: atol=0.1, cos_threshold=0.999
  - Prefix encoder: atol=0.1, cos_threshold=0.997
  - Suffix denoiser (single step): atol=0.15, cos_threshold=0.999

Run with: source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
          python validate_neffs.py [--only vision|prefix|suffix]
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, '/home/ubuntu/pi0-port')
sys.path.insert(0, '/home/ubuntu/pi0-port/skills/scripts')

CHECKPOINT_FILE = "/home/ubuntu/pi0-port/weights/model.safetensors"
COMPILED_DIR = "/home/ubuntu/pi0-port/compiled"


def validate_neff(neff_out, ref_out, name: str, atol: float = 0.1, cos_threshold: float = 0.99):
    neff_f = neff_out.float().flatten()
    ref_f = ref_out.float().flatten()
    mean_diff = (neff_f - ref_f).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(neff_f, ref_f, dim=0).item()
    print(f"{name}: mean_diff={mean_diff:.4f}  cos_sim={cos_sim:.6f}")
    assert mean_diff < atol, (
        f"{name} FAILED: mean_diff={mean_diff:.4f} > {atol}. "
        "If mean_diff > 1.0, weights were NOT loaded before trace — fix load_module()."
    )
    assert cos_sim > cos_threshold, f"{name} FAILED: cos_sim={cos_sim:.6f} < {cos_threshold}"
    print(f"{name}: PASSED")
    return mean_diff, cos_sim


def validate_vision():
    import torch_neuronx
    from vision_encoder import load_vision_encoder, NeuronVisionEncoder

    neff_path = os.path.join(COMPILED_DIR, "vision_encoder/model.pt")
    assert os.path.exists(neff_path), f"NEFF not found: {neff_path} — run compile_all.py first"

    print("[vision] Loading CPU reference...")
    ref_encoder = load_vision_encoder(CHECKPOINT_FILE)
    ref_encoder.eval()

    print("[vision] Loading NEFF...")
    neff = torch.jit.load(neff_path)

    torch.manual_seed(0)
    px = torch.randn(1, 3, 224, 224, dtype=torch.float32) * 0.3

    with torch.no_grad():
        ref_out = ref_encoder(px)
        neff_out = neff(px)

    validate_neff(neff_out, ref_out, "vision_encoder", atol=0.1, cos_threshold=0.999)


def validate_prefix():
    import torch_neuronx
    from prefix_encoder import load_prefix_encoder
    from config_constants import PREFIX_LEN, VLM_HIDDEN_SIZE

    neff_path = os.path.join(COMPILED_DIR, "prefix_encoder/model.pt")
    assert os.path.exists(neff_path), f"NEFF not found: {neff_path}"

    print("[prefix] Loading CPU reference...")
    ref_encoder = load_prefix_encoder(CHECKPOINT_FILE, seq_len=PREFIX_LEN)
    ref_encoder.eval()

    print("[prefix] Loading NEFF...")
    neff = torch.jit.load(neff_path)

    torch.manual_seed(0)
    prefix_embs = torch.randn(1, PREFIX_LEN, VLM_HIDDEN_SIZE, dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        ref_kv = ref_encoder(prefix_embs)
        neff_kv = neff(prefix_embs)

    # Validate on just the KV values (flatten K and V separately)
    validate_neff(neff_kv[:, 0], ref_kv[:, 0], "prefix_encoder_K", atol=0.1, cos_threshold=0.997)
    validate_neff(neff_kv[:, 1], ref_kv[:, 1], "prefix_encoder_V", atol=0.1, cos_threshold=0.997)


def validate_suffix():
    from suffix_denoiser import load_suffix_denoiser, NeuronPi0ActionHead
    import torch_neuronx
    from config_constants import PREFIX_LEN, EXPERT_NUM_LAYERS, EXPERT_HEAD_DIM, EXPERT_NUM_KV_HEADS
    import torch_neuronx

    neff_path = os.path.join(COMPILED_DIR, "suffix_denoiser/model.pt")
    assert os.path.exists(neff_path), f"NEFF not found: {neff_path}"

    print("[suffix] Loading CPU reference...")
    ref_wrapper = load_suffix_denoiser(CHECKPOINT_FILE)
    ref_wrapper.eval()

    # ModelBuilder NEFFs require sharded weight initialization after torch.jit.load.
    # Use NeuronPi0ActionHead.load() which calls nxd_model.initialize(sharded_weights).
    print("[suffix] Loading NEFF + sharded weights...")
    suf_path = os.path.join(COMPILED_DIR, "suffix_denoiser/")
    head = NeuronPi0ActionHead(model_path=os.path.dirname(CHECKPOINT_FILE))
    head.load(compiled_model_path=suf_path, skip_warmup=True)
    # head.traced_model is the initialized NEFF ScriptModule.
    # Call it directly — do NOT go through the Python forward (which runs CPU Gemma layers).

    torch.manual_seed(0)
    B = 1
    state = torch.randn(B, 32, dtype=torch.bfloat16) * 0.1
    noisy = torch.randn(B, 50, 32, dtype=torch.bfloat16) * 0.5
    time_emb = torch.randn(B, 1024, dtype=torch.bfloat16) * 0.1
    prefix_kv = torch.randn(EXPERT_NUM_LAYERS, 2, B, PREFIX_LEN, EXPERT_NUM_KV_HEADS,
                            EXPERT_HEAD_DIM, dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        ref_vt = ref_wrapper(state, noisy, time_emb, prefix_kv)
        neff_vt = head.traced_model(state, noisy, time_emb, prefix_kv)

    validate_neff(neff_vt, ref_vt, "suffix_denoiser", atol=0.15, cos_threshold=0.999)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["vision", "prefix", "suffix"], default=None)
    args = parser.parse_args()

    if args.only is None or args.only == "vision":
        validate_vision()
    if args.only is None or args.only == "prefix":
        validate_prefix()
    if args.only is None or args.only == "suffix":
        validate_suffix()

    print("\n=== All NEFF validations PASSED ===")
