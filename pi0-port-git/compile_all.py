"""
compile_all.py — Compile all π0 subgraphs to NEFFs on Trainium.

Run with: source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
          python compile_all.py [--only vision|prefix|suffix] [--force]
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, '/home/ubuntu/pi0-port')
sys.path.insert(0, '/home/ubuntu/pi0-port/skills/scripts')

CHECKPOINT_FILE = "/home/ubuntu/pi0-port/weights/model.safetensors"
COMPILED_DIR = "/home/ubuntu/pi0-port/compiled"


def compile_vision(force: bool = False):
    from vision_encoder import load_vision_encoder, compile_vision_encoder

    save_path = os.path.join(COMPILED_DIR, "vision_encoder/")
    neff_path = os.path.join(save_path, "model.pt")
    if os.path.exists(neff_path) and not force:
        print(f"[vision] Already compiled: {neff_path}")
        return

    print("[vision] Loading weights from checkpoint...")
    encoder = load_vision_encoder(CHECKPOINT_FILE)
    encoder.eval()

    # CPU forward pass before compile (catch errors cheaply)
    example = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        out = encoder(example)
    assert out.shape == (1, 256, 2048), f"CPU forward failed: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in CPU forward"
    print(f"[vision] CPU forward PASSED — output {out.shape}")

    print("[vision] Compiling to NEFF...")
    compile_vision_encoder(encoder, save_path)
    print(f"[vision] DONE: {neff_path}")


def compile_prefix(force: bool = False):
    from prefix_encoder import load_prefix_encoder, compile_prefix_encoder
    from config_constants import PREFIX_LEN, VLM_HIDDEN_SIZE

    save_path = os.path.join(COMPILED_DIR, "prefix_encoder/")
    neff_path = os.path.join(save_path, "model.pt")
    if os.path.exists(neff_path) and not force:
        print(f"[prefix] Already compiled: {neff_path}")
        return

    print("[prefix] Loading weights from checkpoint...")
    encoder = load_prefix_encoder(CHECKPOINT_FILE, seq_len=PREFIX_LEN)
    encoder.eval()

    # CPU forward pass
    example = torch.zeros(1, PREFIX_LEN, VLM_HIDDEN_SIZE, dtype=torch.bfloat16)
    with torch.no_grad():
        kv = encoder(example)
    assert kv.shape[0] == 18, f"CPU forward KV shape: {kv.shape}"
    assert not torch.isnan(kv).any(), "NaN in CPU forward"
    print(f"[prefix] CPU forward PASSED — prefix_kv {kv.shape}")

    print("[prefix] Compiling to NEFF...")
    compile_prefix_encoder(encoder, save_path)
    print(f"[prefix] DONE: {neff_path}")


def compile_suffix(force: bool = False):
    from suffix_denoiser import NeuronPi0ActionHead, load_suffix_denoiser

    save_path = os.path.join(COMPILED_DIR, "suffix_denoiser/")
    neff_path = os.path.join(save_path, "model.pt")
    if os.path.exists(neff_path) and not force:
        print(f"[suffix] Already compiled: {neff_path}")
        return

    print("[suffix] Setting up action head...")
    # NeuronActionHeadBase.get_state_dict expects a DIRECTORY (not file) — it globs *.safetensors inside.
    ckpt_dir = os.path.dirname(CHECKPOINT_FILE)
    head = NeuronPi0ActionHead(model_path=ckpt_dir)

    # compile_denoiser calls _build_denoising_wrapper() (which calls load_module + loads weights)
    # then pre_compile_validate() (CPU forward), then ModelBuilder trace + weight sharding.
    print("[suffix] Compiling to NEFF via ModelBuilder...")
    head.compile_denoiser(save_path=save_path)
    print(f"[suffix] DONE: {neff_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["vision", "prefix", "suffix"], default=None)
    parser.add_argument("--force", action="store_true", help="Recompile even if NEFF exists")
    args = parser.parse_args()

    os.makedirs(COMPILED_DIR, exist_ok=True)

    if args.only is None or args.only == "vision":
        compile_vision(force=args.force)

    if args.only is None or args.only == "prefix":
        compile_prefix(force=args.force)

    if args.only is None or args.only == "suffix":
        compile_suffix(force=args.force)

    print("\n=== All compilations complete ===")
