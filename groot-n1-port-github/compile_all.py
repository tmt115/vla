"""
compile_all.py — Phase 5: Compile all GR00T N1.7 subgraphs to Trainium NEFFs.

All three compiled subgraphs use NxDI ModelBuilder (TP=8 via parallel_state):
  1. Backbone LLM (16 Qwen3-VL layers)    — NeuronGR00TBackboneHead.compile_denoiser()
  2. VL self-attention (4 layers)          — wrapper via NeuronActionHeadBase.compile_denoiser()
  3. DiT action head (32 layers)           — NeuronGR00TActionHead.compile_denoiser()

ViT runs on CPU (dynamic shape ops not supported by neuronx-cc).

CRITICAL: All wrappers use lazy model init (load_module pattern). This ensures
ColumnParallelLinear/RowParallelLinear use real TP=8 after parallel_state is
initialized by ModelBuilder — not the nn.Linear fallback.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/groot-n1-port
    python compile_all.py 2>&1 | tee compile_all.log
"""

import os
import sys
import glob
import time
import shutil
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'skills', 'scripts'))
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

from safetensors.torch import load_file

from config_constants import (
    BATCH_SIZE, MODEL_PATH, TP_DEGREE_VLM, TP_DEGREE_DIT,
    NUM_CONDITIONING_TOKENS, LLM_HIDDEN_SIZE, VL_SELF_ATTN_HIDDEN_SIZE,
    DIT_INPUT_SEQ_LEN, DIT_OUTPUT_DIM, TIMESTEP_EMBED_DIM,
    CONDITIONING_HIDDEN_SIZE,
)
from vlm_backbone_block import (
    NeuronGR00TBackboneHead,
    load_backbone_state_dict,
    _make_backbone_compile_config,
)
from vl_self_attention_block import (
    NeuronGR00TVLSelfAttnWrapper,
    load_vl_self_attn_state_dict,
)
from dit_block import (
    NeuronGR00TActionHead,
    load_dit_state_dict,
)
from neuron_action_head_base import NeuronActionHeadBase, NeuronDenoisingConfig

COMPILED_DIR = os.path.join(ROOT, 'compiled')
BACKBONE_DIR = os.path.join(COMPILED_DIR, 'backbone')
VL_SA_DIR    = os.path.join(COMPILED_DIR, 'vl_self_attn')
DIT_DIR      = os.path.join(COMPILED_DIR, 'dit')

MODEL_FILE = 'model.pt'


def _is_compiled(path: str) -> bool:
    return os.path.isfile(os.path.join(path, MODEL_FILE))


def _load_checkpoint() -> dict:
    """Load full GR00T checkpoint from safetensors shards."""
    print("Loading checkpoint...")
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(MODEL_PATH, '*.safetensors')))
    if not files:
        raise FileNotFoundError(f"No safetensors in {MODEL_PATH}")
    sd = {}
    for f in files:
        sd.update(load_file(f))
    print(f"  Loaded {len(sd)} tensors in {time.time()-t0:.1f}s")
    return sd


# ---------------------------------------------------------------------------
# Backbone compile
# ---------------------------------------------------------------------------

def compile_backbone(hf_sd: dict, force: bool = False) -> None:
    save_path = BACKBONE_DIR + '/'
    if _is_compiled(BACKBONE_DIR) and not force:
        print(f"[backbone] Already compiled at {BACKBONE_DIR}, skipping.")
        return

    print("[backbone] Building state dict...")
    backbone_sd = load_backbone_state_dict(hf_sd)
    print(f"  {len(backbone_sd)} weight tensors")

    cfg = _make_backbone_compile_config()
    head = NeuronGR00TBackboneHead(
        model_path=MODEL_PATH,
        config=cfg,
        state_dict=backbone_sd,
    )

    print(f"[backbone] Compiling to {save_path} ...")
    t0 = time.time()
    # compile_denoiser() calls _build_denoising_wrapper() + pre_compile_validate() + trace().
    # Wrappers auto-call load_module() in forward() if model is None.
    head.compile_denoiser(save_path)
    print(f"[backbone] Done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# VL self-attention compile
# ---------------------------------------------------------------------------

def _make_vl_self_attn_compile_config() -> NeuronDenoisingConfig:
    return NeuronDenoisingConfig(
        batch_size=BATCH_SIZE,
        tp_degree=TP_DEGREE_VLM,
        action_chunk_size=NUM_CONDITIONING_TOKENS,
        action_dim=VL_SELF_ATTN_HIDDEN_SIZE,
        num_conditioning_tokens=NUM_CONDITIONING_TOKENS,
        conditioning_hidden_size=VL_SELF_ATTN_HIDDEN_SIZE,
        timestep_embed_dim=VL_SELF_ATTN_HIDDEN_SIZE,
    )


class _VLSelfAttnHead(NeuronActionHeadBase):
    """Thin head class for VL self-attention compilation."""

    def __init__(self, model_path, config, state_dict=None):
        super().__init__(model_path, config)
        self._preload_sd = state_dict

    def _build_denoising_wrapper(self):
        wrapper = NeuronGR00TVLSelfAttnWrapper()
        wrapper._preload_sd = self._preload_sd
        return wrapper

    def _build_attention_mask(self):
        return None  # VL self-attn uses internal full attention

    def checkpoint_loader_fn(self, mmap=False):
        return self._preload_sd if self._preload_sd else {}

    def get_compiler_args(self):
        return "-O1"

    def get_conditioning_contract(self):
        from neuron_action_head_base import ConditioningContract
        return ConditioningContract(
            num_conditioning_tokens=NUM_CONDITIONING_TOKENS,
            conditioning_hidden_size=VL_SELF_ATTN_HIDDEN_SIZE,
        )

    def _get_timestep_sequence(self, num_steps):
        return []

    def _embed_timestep(self, t):
        return torch.zeros(BATCH_SIZE, VL_SELF_ATTN_HIDDEN_SIZE, dtype=torch.bfloat16)


def compile_vl_self_attn(hf_sd: dict, force: bool = False) -> None:
    save_path = VL_SA_DIR + '/'
    if _is_compiled(VL_SA_DIR) and not force:
        print(f"[vl_self_attn] Already compiled at {VL_SA_DIR}, skipping.")
        return

    print("[vl_self_attn] Building state dict...")
    vl_sd = load_vl_self_attn_state_dict(hf_sd)
    print(f"  {len(vl_sd)} weight tensors")

    cfg = _make_vl_self_attn_compile_config()
    head = _VLSelfAttnHead(
        model_path=MODEL_PATH,
        config=cfg,
        state_dict=vl_sd,
    )

    print(f"[vl_self_attn] Compiling to {save_path} ...")
    t0 = time.time()
    head.compile_denoiser(save_path)
    print(f"[vl_self_attn] Done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# DiT compile
# ---------------------------------------------------------------------------

def _make_dit_compile_config() -> NeuronDenoisingConfig:
    return NeuronDenoisingConfig(
        batch_size=BATCH_SIZE,
        tp_degree=TP_DEGREE_DIT,
        action_chunk_size=DIT_INPUT_SEQ_LEN,
        action_dim=DIT_OUTPUT_DIM,
        num_conditioning_tokens=NUM_CONDITIONING_TOKENS,
        conditioning_hidden_size=CONDITIONING_HIDDEN_SIZE,
        timestep_embed_dim=TIMESTEP_EMBED_DIM,
    )


def compile_dit(hf_sd: dict, force: bool = False) -> None:
    save_path = DIT_DIR + '/'
    if _is_compiled(DIT_DIR) and not force:
        print(f"[dit] Already compiled at {DIT_DIR}, skipping.")
        return

    print("[dit] Building state dict...")
    dit_sd = load_dit_state_dict(hf_sd)
    print(f"  {len(dit_sd)} weight tensors")

    cfg = _make_dit_compile_config()
    head = NeuronGR00TActionHead(
        model_path=MODEL_PATH,
        config=cfg,
        state_dict=dit_sd,
    )

    print(f"[dit] Compiling to {save_path} ...")
    t0 = time.time()
    head.compile_denoiser(save_path)
    print(f"[dit] Done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Pre-compile CPU validation helper
# ---------------------------------------------------------------------------

def _validate_with_load_module(head: NeuronActionHeadBase) -> None:
    """
    Run CPU forward pass validation before handing off to ModelBuilder.

    compile_denoiser() calls pre_compile_validate() before trace(). Since
    load_module() is only called by ModelBuilder (after parallel_state init),
    the wrapper's model is None during pre_compile_validate(). We patch it:
    1. Build wrapper manually
    2. Call load_module() (CPU path: tp=1, LlamaRMSNorm)
    3. Run pre_compile_validate()
    4. Reset wrapper (compile_denoiser will rebuild it via _build_denoising_wrapper)
    """
    wrapper = head._build_denoising_wrapper()
    wrapper.load_module()
    head.denoising_wrapper = wrapper
    head.attention_mask = head._build_attention_mask()
    head.pre_compile_validate()
    # Reset so compile_denoiser() rebuilds from scratch with correct TP
    head.denoising_wrapper = None
    head.attention_mask = None


# ---------------------------------------------------------------------------
# Pre-compile TP verification
# ---------------------------------------------------------------------------

def _verify_tp(wrapper, name: str) -> None:
    """Verify that ColumnParallelLinear is used (not nn.Linear fallback)."""
    try:
        from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
        from neuronx_distributed.parallel_layers import parallel_state
        if parallel_state.model_parallel_is_initialized():
            has_parallel = any(
                isinstance(m, ColumnParallelLinear) for m in wrapper.model.modules()
            )
            assert has_parallel, f"TP FAILED for {name} — ColumnParallelLinear not found"
            print(f"[{name}] TP=8 verification PASSED (ColumnParallelLinear found)")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compile GR00T N1.7 subgraphs to NEFFs')
    parser.add_argument('--force', action='store_true', help='Recompile even if already compiled')
    parser.add_argument('--only', choices=['backbone', 'vl_self_attn', 'dit'], help='Compile only one subgraph')
    args = parser.parse_args()

    # Clean broken compiled dirs if force
    if args.force:
        for d in [BACKBONE_DIR, VL_SA_DIR, DIT_DIR]:
            if os.path.exists(d):
                print(f"Removing {d}")
                shutil.rmtree(d)

    os.makedirs(BACKBONE_DIR, exist_ok=True)
    os.makedirs(VL_SA_DIR, exist_ok=True)
    os.makedirs(DIT_DIR, exist_ok=True)

    hf_sd = _load_checkpoint()

    t_total = time.time()

    if args.only == 'backbone' or args.only is None:
        compile_backbone(hf_sd, force=args.force)

    if args.only == 'vl_self_attn' or args.only is None:
        compile_vl_self_attn(hf_sd, force=args.force)

    if args.only == 'dit' or args.only is None:
        compile_dit(hf_sd, force=args.force)

    print(f"\nAll compilations done in {(time.time()-t_total)/60:.1f} min")
    print("Check compiled/ for model.pt files.")


if __name__ == '__main__':
    main()
