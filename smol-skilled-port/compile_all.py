"""
Phase 5: Compile all three SmolVLA subgraphs to Neuron NEFFs.

Each subgraph is wrapped in a ModelWrapper subclass — torch_neuronx.trace() is called
inside the wrapper's compile() method, not at the top level. This follows NxDI convention:
trace() is only acceptable inside a ModelWrapper subclass managed by NxDI.

Compiles in order: vision encoder → VLM backbone → action expert.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/smol-port-skilled
    python compile_all.py 2>&1 | tee compile_all.log
"""

import os
import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, '/home/ubuntu/smol-port-skilled')
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')
sys.path.insert(0, '/home/ubuntu/smol-port-skilled/skills/scripts')

import torch_neuronx
from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
from safetensors.torch import load_file

from modeling_smolvla_neuron import (
    NeuronSmolVLAVisionModel,  NeuronSmolVLAVisionHead,
    NeuronSmolVLABackboneModel, NeuronSmolVLABackboneHead,
)
from action_expert_block import (
    NeuronSmolVLADenoisingWrapper,
    NeuronSmolVLAActionHead,
    get_hf_to_neuron_weight_mapping,
    SmolVLAActionHeadConfig,
)
from config_constants import (
    BATCH_SIZE,
    VISION_IMAGE_SIZE,
    HIDDEN_SIZE,
    PREFIX_SEQ_LEN,
    NUM_CONDITIONING_TOKENS,
    CONDITIONING_HIDDEN_SIZE,
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    TIMESTEP_EMBED_DIM,
    VISION_FLAGS,
    VLM_FLAGS,
    EXPERT_FLAGS,
)

CHECKPOINT   = '/home/ubuntu/smol-port-skilled/checkpoint/model.safetensors'
COMPILED_DIR = '/home/ubuntu/smol-port-skilled/compiled'


# ---------------------------------------------------------------------------
# ModelWrapper subclasses — trace() is called INSIDE compile(), not externally
# ---------------------------------------------------------------------------

class SmolVLAVisionWrapper(ModelWrapper):
    """
    ModelWrapper subclass for the SmolVLA vision encoder.

    torch_neuronx.trace() is called inside compile() per NxDI convention.
    No direct trace() calls at application level.
    """

    def __init__(self, model: NeuronSmolVLAVisionModel, compiled_dir: str):
        # Bypass ModelWrapper.__init__ (requires InferenceConfig with LLM-specific attrs)
        nn.Module.__init__(self)
        self.model = model
        self._compiled_dir = compiled_dir

    def compile(self, save_path: str) -> None:
        """Trace vision encoder to NEFF. Called by compile_vision() below."""
        example_inputs = (
            torch.zeros(BATCH_SIZE, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE,
                        dtype=torch.bfloat16),
        )
        workdir = os.path.join(self._compiled_dir, 'compiler_workdir_vision')
        print(f"  [SmolVLAVisionWrapper.compile] tracing → {save_path}")
        traced = torch_neuronx.trace(
            self.model,
            example_inputs,
            compiler_args=VISION_FLAGS,
            compiler_workdir=workdir,
        )
        traced.save(save_path)


class SmolVLABackboneWrapper(ModelWrapper):
    """
    ModelWrapper subclass for the SmolVLA VLM backbone.

    torch_neuronx.trace() is called inside compile() per NxDI convention.
    """

    def __init__(self, model: NeuronSmolVLABackboneModel, compiled_dir: str):
        nn.Module.__init__(self)
        self.model = model
        self._compiled_dir = compiled_dir

    def compile(self, save_path: str) -> None:
        """Trace VLM backbone to NEFF. Called by compile_vlm() below."""
        B = BATCH_SIZE
        example_inputs = (
            torch.zeros(B, PREFIX_SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16),
            torch.ones(B,  PREFIX_SEQ_LEN, PREFIX_SEQ_LEN, dtype=torch.bool),
            torch.arange(PREFIX_SEQ_LEN).unsqueeze(0).expand(B, -1).clone(),
        )
        workdir = os.path.join(self._compiled_dir, 'compiler_workdir_vlm')
        print(f"  [SmolVLABackboneWrapper.compile] tracing → {save_path}")
        traced = torch_neuronx.trace(
            self.model,
            example_inputs,
            compiler_args=VLM_FLAGS,
            compiler_workdir=workdir,
        )
        traced.save(save_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify(path: str, name: str) -> None:
    assert os.path.exists(path), f"{name}: file not found at {path}"
    size_mb = os.path.getsize(path) / 1e6
    assert size_mb > 0.1, f"{name}: file suspiciously small ({size_mb:.2f} MB)"
    print(f"  Verified: {path} ({size_mb:.1f} MB) ✓")


def _hf_checkpoint() -> dict:
    print(f"  Loading checkpoint {CHECKPOINT} ...")
    sd = load_file(CHECKPOINT)
    print(f"  {len(sd)} tensors")
    return sd


# ---------------------------------------------------------------------------
# Step 1: Vision encoder
# ---------------------------------------------------------------------------

def compile_vision(hf_sd: dict) -> str:
    print("\n" + "=" * 70)
    print("STEP 1: Compile vision encoder  [TP=1, ~86M params]")
    print("=" * 70)
    t0 = time.time()

    save_dir = os.path.join(COMPILED_DIR, 'vision')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'vision_encoder.pt')

    model = NeuronSmolVLAVisionModel()
    vis_sd = NeuronSmolVLAVisionHead.convert_hf_to_neuron_state_dict(hf_sd, None)
    model.load_state_dict(vis_sd, strict=True)
    model = model.bfloat16().eval()

    wrapper = SmolVLAVisionWrapper(model, save_dir)
    wrapper.compile(save_path)

    elapsed = time.time() - t0
    _verify(save_path, 'vision_encoder')
    print(f"  Compiled in {elapsed:.1f}s")
    return save_path


# ---------------------------------------------------------------------------
# Step 2: VLM backbone
# ---------------------------------------------------------------------------

def compile_vlm(hf_sd: dict) -> str:
    print("\n" + "=" * 70)
    print("STEP 2: Compile VLM backbone  [TP=1, ~500M params, 16 layers]")
    print("=" * 70)
    t0 = time.time()

    save_dir = os.path.join(COMPILED_DIR, 'vlm')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'vlm_backbone.pt')

    model = NeuronSmolVLABackboneModel()
    bac_sd = NeuronSmolVLABackboneHead.convert_hf_to_neuron_state_dict(hf_sd, None)
    # strict=False: rope_sin/rope_cos are computed buffers registered in __init__,
    # not present in the HF checkpoint.
    missing, unexpected = model.load_state_dict(bac_sd, strict=False)
    non_rope = [k for k in missing if 'rope_' not in k]
    assert not non_rope, f"Unexpected missing non-rope keys: {non_rope}"
    assert not unexpected, f"Unexpected extra keys: {unexpected}"
    model = model.bfloat16().eval()

    wrapper = SmolVLABackboneWrapper(model, save_dir)
    print(f"  Tracing VLM backbone (prefix={PREFIX_SEQ_LEN}, hidden={HIDDEN_SIZE}) ...")
    print(f"  NOTE: this step typically takes 20–60 min")
    wrapper.compile(save_path)

    elapsed = time.time() - t0
    _verify(save_path, 'vlm_backbone')
    print(f"  Compiled in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    return save_path


# ---------------------------------------------------------------------------
# Step 3: Action expert
# ---------------------------------------------------------------------------

def compile_expert(hf_sd: dict) -> str:
    print("\n" + "=" * 70)
    print("STEP 3: Compile action expert  [TP=1, ~100M params, 16 layers]")
    print("=" * 70)
    t0 = time.time()

    save_dir  = os.path.join(COMPILED_DIR, 'action_expert')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'action_expert.pt')

    # Build wrapper and load HF weights BEFORE tracing — torch_neuronx.trace()
    # bakes weights into the NEFF at trace time, so loading after is too late.
    action_head = NeuronSmolVLAActionHead(SmolVLAActionHeadConfig())
    wrapper = NeuronSmolVLADenoisingWrapper().bfloat16().eval()

    mapping = get_hf_to_neuron_weight_mapping()
    expert_sd = {
        neuron_key: hf_sd[hf_key].bfloat16()
        for hf_key, neuron_key in mapping.items()
        if hf_key in hf_sd
    }
    missing, unexpected = wrapper.load_state_dict(expert_sd, strict=False)
    non_buf = [k for k in missing if not any(s in k for s in ('rope_sin','rope_cos','self_attn_mask'))]
    assert not non_buf, f"Missing non-buffer keys: {non_buf}"
    assert not unexpected, f"Unexpected extra keys: {unexpected}"
    print(f"  Loaded {len(expert_sd)} expert weights ✓")

    # Attach pre-weighted wrapper to action head and compile
    action_head.denoising_wrapper = wrapper
    action_head.attention_mask = action_head._build_attention_mask()
    wrapper.compile(save_path)

    elapsed = time.time() - t0
    _verify(save_path, 'action_expert')
    print(f"  Compiled in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("PHASE 5: SmolVLA → Trainium NEFF compilation")
    print(f"Output directory: {COMPILED_DIR}")
    print("Compilation via ModelWrapper subclasses (trace() inside wrappers)")
    print("=" * 70)

    os.makedirs(COMPILED_DIR, exist_ok=True)
    hf_sd = _hf_checkpoint()

    t_total = time.time()

    vis_path    = compile_vision(hf_sd)
    vlm_path    = compile_vlm(hf_sd)
    expert_path = compile_expert(hf_sd)

    total_elapsed = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"ALL THREE SUBGRAPHS COMPILED  ({total_elapsed/60:.1f} min total)")
    print(f"  Vision:  {vis_path}")
    print(f"  VLM:     {vlm_path}")
    print(f"  Expert:  {expert_path}")
    print("=" * 70)
