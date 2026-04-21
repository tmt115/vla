"""
Phase 3 smoke test: verify NxDI scaffolding classes instantiate and wire correctly.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/smol-port-skilled
    python tests/test_phase3_scaffolding.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEROBOT_PATH = '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages'
if LEROBOT_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_PATH)

import torch

from config_constants import (
    BATCH_SIZE, VISION_IMAGE_SIZE, HIDDEN_SIZE, PREFIX_SEQ_LEN,
    NUM_TEXT_TOKENS, NUM_STATE_TOKENS, VISION_TOKENS_PER_CAMERA,
    ACTION_CHUNK_SIZE, ACTION_DIM, NUM_CONDITIONING_TOKENS,
    CONDITIONING_HIDDEN_SIZE, NUM_DENOISING_STEPS,
)
from modeling_smolvla_neuron import (
    SmolVLANeuronConfig,
    SmolVLAInferenceConfig,
    NeuronSmolVLAVisionModel,
    NeuronSmolVLABackboneModel,
    NeuronSmolVLAVisionHead,
    NeuronSmolVLABackboneHead,
    NeuronSmolVLAActionHead,
    NeuronSmolVLAPolicy,
    _NEURON_AVAILABLE,
)


def test_config_classes():
    print("\n--- Config classes ---")
    neuron_cfg = SmolVLANeuronConfig(
        tp_degree=8,
        tp_degree_vision=1,
        tp_degree_vlm=8,
        tp_degree_expert=2,
    )
    assert neuron_cfg.tp_degree_vision == 1
    assert neuron_cfg.tp_degree_vlm == 8
    assert neuron_cfg.tp_degree_expert == 2
    print(f"  SmolVLANeuronConfig: tp_vision={neuron_cfg.tp_degree_vision}, "
          f"tp_vlm={neuron_cfg.tp_degree_vlm}, tp_expert={neuron_cfg.tp_degree_expert}")

    assert SmolVLAInferenceConfig.get_neuron_config_cls() is SmolVLANeuronConfig
    # Instantiate with all required attributes supplied as kwargs
    inf_cfg = SmolVLAInferenceConfig(
        neuron_config=neuron_cfg,
        hidden_size=960, num_attention_heads=15, num_key_value_heads=5,
        num_hidden_layers=16, intermediate_size=2560, rms_norm_eps=1e-5,
        max_position_embeddings=8192, vocab_size=49152, vision_config=None,
        action_chunk_size=50, action_dim=32, state_dim=32,
        num_denoising_steps=10, min_period=0.001, max_period=1000.0,
        timestep_embed_dim=720,
    )
    attrs = inf_cfg.get_required_attributes()
    assert "hidden_size" in attrs
    assert "action_chunk_size" in attrs
    print(f"  SmolVLAInferenceConfig: {len(attrs)} required attributes ✓")
    print("  PASS")


def test_model_bodies():
    print("\n--- NeuronBaseModel subclasses (CPU instantiation) ---")
    B = BATCH_SIZE

    # Vision model
    vision_model = NeuronSmolVLAVisionModel()
    pixel = torch.randn(B, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)
    with torch.no_grad():
        out = vision_model(pixel)
    assert out.shape == (B, VISION_TOKENS_PER_CAMERA, HIDDEN_SIZE), f"Got {out.shape}"
    print(f"  NeuronSmolVLAVisionModel: {tuple(pixel.shape)} → {tuple(out.shape)} ✓")

    # Backbone model
    backbone_model = NeuronSmolVLABackboneModel()
    prefix = torch.randn(B, PREFIX_SEQ_LEN, HIDDEN_SIZE)
    attn_mask = torch.ones(B, PREFIX_SEQ_LEN, PREFIX_SEQ_LEN, dtype=torch.bool)
    pos_ids = torch.arange(PREFIX_SEQ_LEN).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        cond = backbone_model(prefix, attn_mask, pos_ids)
    assert cond.shape == (B, NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE), f"Got {cond.shape}"
    print(f"  NeuronSmolVLABackboneModel: {tuple(prefix.shape)} → {tuple(cond.shape)} ✓")
    print("  PASS")


def test_application_heads():
    print("\n--- Application head class structure ---")
    # Verify _model_cls, get_config_cls, convert_hf_to_neuron_state_dict are defined
    assert NeuronSmolVLAVisionHead._model_cls is NeuronSmolVLAVisionModel
    assert NeuronSmolVLAVisionHead.get_config_cls() is SmolVLAInferenceConfig
    sd = NeuronSmolVLAVisionHead.convert_hf_to_neuron_state_dict({}, None)
    assert isinstance(sd, dict)
    print("  NeuronSmolVLAVisionHead: _model_cls, get_config_cls, convert_hf_to_neuron_state_dict ✓")

    assert NeuronSmolVLABackboneHead._model_cls is NeuronSmolVLABackboneModel
    assert NeuronSmolVLABackboneHead.get_config_cls() is SmolVLAInferenceConfig
    sd = NeuronSmolVLABackboneHead.convert_hf_to_neuron_state_dict({}, None)
    assert isinstance(sd, dict)
    print("  NeuronSmolVLABackboneHead: _model_cls, get_config_cls, convert_hf_to_neuron_state_dict ✓")
    print("  PASS")


def test_action_head_wiring():
    print("\n--- Action expert head wiring ---")
    head = NeuronSmolVLAActionHead()
    contract = head.get_conditioning_contract()
    assert contract.num_conditioning_tokens == NUM_CONDITIONING_TOKENS
    assert contract.conditioning_hidden_size == CONDITIONING_HIDDEN_SIZE
    print(f"  NeuronSmolVLAActionHead conditioning contract: "
          f"tokens={contract.num_conditioning_tokens}, hidden={contract.conditioning_hidden_size} ✓")

    # compile_denoiser on CPU (no-op compile, but builds denoising_wrapper)
    head.compile_denoiser(save_path="/tmp/test_expert_neff")
    assert head.denoising_wrapper is not None
    assert head.attention_mask is not None
    print(f"  compile_denoiser(): denoising_wrapper built ✓")
    print(f"  attention_mask shape: {tuple(head.attention_mask.shape)} ✓")
    print("  PASS")


def test_policy_instantiation():
    print("\n--- NeuronSmolVLAPolicy instantiation ---")
    policy = NeuronSmolVLAPolicy()
    assert isinstance(policy.vision_model, NeuronSmolVLAVisionModel)
    assert isinstance(policy.backbone_model, NeuronSmolVLABackboneModel)
    assert isinstance(policy.action_head, NeuronSmolVLAActionHead)
    print("  Policy holds all three subgraph heads ✓")
    print("  PASS")


def test_policy_run():
    print("\n--- NeuronSmolVLAPolicy.run() (CPU, random weights) ---")
    torch.manual_seed(42)
    policy = NeuronSmolVLAPolicy()
    policy.compile("/tmp/test_policy_compiled")  # builds denoising_wrapper (no-op on CPU)

    B = BATCH_SIZE
    pixel_values = torch.randn(B, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)
    text_embs    = torch.randn(B, NUM_TEXT_TOKENS, HIDDEN_SIZE)
    state_embs   = torch.randn(B, NUM_STATE_TOKENS, HIDDEN_SIZE)

    actions = policy.run(pixel_values, text_embs, state_embs, num_denoising_steps=NUM_DENOISING_STEPS)
    assert actions.shape == (B, ACTION_CHUNK_SIZE, ACTION_DIM), f"Got {actions.shape}"
    assert not torch.isnan(actions).any()
    assert not torch.isinf(actions).any()
    print(f"  Output shape: {tuple(actions.shape)} ✓")
    print(f"  No NaN/Inf ✓")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3 SCAFFOLDING SMOKE TEST")
    print(f"NxDI available: {_NEURON_AVAILABLE}")
    print("=" * 70)

    test_config_classes()
    test_model_bodies()
    test_application_heads()
    test_action_head_wiring()
    test_policy_instantiation()
    test_policy_run()

    print("\n" + "=" * 70)
    print("ALL PHASE 3 TESTS PASSED")
    print("=" * 70)
