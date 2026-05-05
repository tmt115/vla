"""
run_inference.py — π0 inference on AWS Trainium.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python run_inference.py

API:
    from run_inference import load_model, generate_actions
    model = load_model()
    actions = generate_actions(model, images, lang_tokens, lang_masks, state)
"""

import os
import sys
import math
import torch
import numpy as np

sys.path.insert(0, '/home/ubuntu/pi0-port')
sys.path.insert(0, '/home/ubuntu/pi0-port/skills/scripts')

CHECKPOINT_PATH = "/home/ubuntu/pi0-port/weights"
COMPILED_DIR = "/home/ubuntu/pi0-port/compiled"


def load_model(
    checkpoint_path: str = CHECKPOINT_PATH,
    compiled_dir: str = COMPILED_DIR,
):
    """
    Load compiled π0 NEFFs and checkpoint weights for inference.

    Returns a Pi0NeuronModel instance ready for generate_actions().
    """
    from pi0_neuron import Pi0NeuronModel
    return Pi0NeuronModel.load(
        checkpoint_path=checkpoint_path,
        compiled_dir=compiled_dir,
    )


def generate_actions(
    model,
    images: list,              # list of 3 tensors [1, 3, 224, 224] float32 in [-1, 1]
    lang_tokens: torch.Tensor, # [1, 48] int64
    lang_masks: torch.Tensor,  # [1, 48] bool
    state: torch.Tensor,       # [1, 6] float32
    num_steps: int = 10,
) -> np.ndarray:               # [50, 6] float32 action chunk
    """
    Generate a 50-step action chunk from π0.

    Args:
        model:      Pi0NeuronModel loaded via load_model()
        images:     List of 3 preprocessed camera images [1, 3, 224, 224] in [-1, 1]
        lang_tokens: Tokenized language instruction [1, 48] (padded)
        lang_masks:  Attention mask for language tokens [1, 48]
        state:       Robot joint state [1, 6]
        num_steps:   Number of Euler denoising steps (default: 10)

    Returns:
        actions: numpy array [50, 6] — 50 steps × 6 joints
    """
    model.num_inference_steps = num_steps
    with torch.no_grad():
        actions = model.generate_actions(images, lang_tokens, lang_masks, state)
    return actions[0].cpu().numpy()  # [50, 6]


if __name__ == "__main__":
    print("=== π0 Trainium Inference Test ===\n")

    # ── Verify compiled NEFFs exist ────────────────────────────────────────
    for subgraph in ["vision_encoder", "prefix_encoder", "suffix_denoiser"]:
        path = os.path.join(COMPILED_DIR, subgraph, "model.pt")
        if not os.path.exists(path):
            print(f"ERROR: {path} not found — run compile_all.py first")
            sys.exit(1)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {subgraph}: {size_mb:.1f} MB")

    print("\nLoading compiled model...")
    model = load_model()
    print("Model loaded.\n")

    # ── Dummy inputs ───────────────────────────────────────────────────────
    # 3 camera images: [1, 3, 224, 224] float32 in [-1, 1]
    images = [
        torch.randn(1, 3, 224, 224, dtype=torch.float32) * 0.5
        for _ in range(3)
    ]
    # Language tokens: dummy (zeros = padding token)
    lang_tokens = torch.zeros(1, 48, dtype=torch.long)
    lang_masks = torch.ones(1, 48, dtype=torch.bool)
    # Robot state: dummy 6-DOF
    state = torch.zeros(1, 6, dtype=torch.float32)

    print("Running inference with dummy inputs...")
    import time
    start = time.perf_counter()
    actions = generate_actions(model, images, lang_tokens, lang_masks, state)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\nOutput shape: {actions.shape}")
    print(f"Output dtype: {actions.dtype}")
    print(f"Output range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"Any NaN: {np.isnan(actions).any()}")
    print(f"Latency: {elapsed_ms:.1f} ms")

    assert actions.shape == (50, 6), f"Expected (50, 6), got {actions.shape}"
    assert not np.isnan(actions).any(), "NaN in output actions"
    print("\n=== run_inference.py: PASSED ===")
