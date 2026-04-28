"""
tests/test_dit_correctness.py

Numerical correctness test: GR00TDiTModel vs original AlternateVLDiT from Isaac-GR00T.
Loads HF weights into both models and compares outputs on identical random inputs.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/groot-n1-port
    python -m pytest tests/test_dit_correctness.py -v -s
"""
import sys, os, glob
sys.path.insert(0, '/home/ubuntu/groot-n1-port')
sys.path.insert(0, '/home/ubuntu/groot-port/skills/scripts')
sys.path.insert(0, '/home/ubuntu/Isaac-GR00T')

import torch
import pytest
from safetensors.torch import load_file

from config_constants import (
    BATCH_SIZE, DIT_INPUT_SEQ_LEN, DIT_HIDDEN_SIZE, DIT_OUTPUT_DIM,
    NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE, TIMESTEP_EMBED_DIM,
    DIT_NUM_LAYERS, DIT_CROSS_ATTN_DIM,
)
from dit_block import GR00TDiTModel, load_dit_model_state_dict

MODEL_PATH = '/home/ubuntu/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495'


@pytest.fixture(scope="module")
def hf_sd():
    sd = {}
    for f in sorted(glob.glob(os.path.join(MODEL_PATH, '*.safetensors'))):
        sd.update(load_file(f))
    return sd


@pytest.fixture(scope="module")
def neuron_model(hf_sd):
    m = GR00TDiTModel().bfloat16().eval()
    dit_sd = load_dit_model_state_dict(hf_sd)
    missing, unexpected = m.load_state_dict(dit_sd, strict=True)
    assert not missing, f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    return m


@pytest.fixture(scope="module")
def ref_model(hf_sd):
    """Original AlternateVLDiT with HF weights loaded directly."""
    from gr00t.model.modules.dit import AlternateVLDiT
    import json

    with open(os.path.join(MODEL_PATH, 'config.json')) as f:
        cfg = json.load(f)

    dit_cfg = cfg['diffusion_model_cfg']
    model = AlternateVLDiT(
        **dit_cfg,
        cross_attention_dim=cfg['backbone_embedding_dim'],
        attend_text_every_n_blocks=cfg.get('attend_text_every_n_blocks', 2),
    ).bfloat16().eval()

    # Load HF weights (prefix: action_head.model.)
    prefix = 'action_head.model.'
    ref_sd = {k[len(prefix):]: v for k, v in hf_sd.items() if k.startswith(prefix)}
    model.load_state_dict(ref_sd, strict=False)
    return model


def _report(name, ref_out, neuron_out):
    diff = (ref_out.float() - neuron_out.float()).abs()
    mean_diff = diff.mean().item()
    max_diff  = diff.max().item()
    a = ref_out.float().flatten()
    b = neuron_out.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    print(f"\n  {name}: mean_diff={mean_diff:.5f}  max_diff={max_diff:.4f}  cos_sim={cos:.6f}")
    return mean_diff, cos


def test_dit_correctness(neuron_model, ref_model):
    """GR00TDiTModel output must match AlternateVLDiT within bfloat16 tolerance."""
    torch.manual_seed(0)
    B = BATCH_SIZE

    # Inputs that match the neuron model's interface
    sa_embs  = torch.randn(B, DIT_INPUT_SEQ_LEN,       DIT_HIDDEN_SIZE,          dtype=torch.bfloat16)
    cond     = torch.randn(B, NUM_CONDITIONING_TOKENS,  CONDITIONING_HIDDEN_SIZE, dtype=torch.bfloat16)
    temb     = torch.randn(B, TIMESTEP_EMBED_DIM,                                 dtype=torch.bfloat16)
    cross_mask = torch.ones(B, 1, DIT_INPUT_SEQ_LEN, NUM_CONDITIONING_TOKENS, dtype=torch.int32)

    # Reference model takes different args: hidden_states, encoder_hidden_states, timestep, ...
    # timestep must be a long tensor (discrete bucket index), temb is produced inside
    # We need to align the two interfaces.
    #
    # Deviation: GR00TDiTModel accepts the projected timestep embedding directly.
    # AlternateVLDiT computes the embedding internally from a scalar timestep.
    # To compare apples-to-apples, we extract the TimestepEncoder from the ref model,
    # run it to get temb, then pass the same temb to our model.
    t_scalar = torch.tensor([500], dtype=torch.long)  # midpoint timestep bucket

    with torch.no_grad():
        # Get temb from ref model's TimestepEncoder
        ref_temb = ref_model.timestep_encoder(t_scalar).bfloat16()  # [1, 1536]

        # Reference forward (AlternateVLDiT requires image_mask + backbone_attention_mask)
        # image_mask: True for image token positions. For simplicity use all-False
        # (all tokens treated as text). This matches our deviation of attending to all tokens.
        image_mask = torch.zeros(B, NUM_CONDITIONING_TOKENS, dtype=torch.bool)
        backbone_attn_mask = torch.ones(B, NUM_CONDITIONING_TOKENS, dtype=torch.bool)

        ref_out = ref_model(
            hidden_states=sa_embs,
            encoder_hidden_states=cond,
            timestep=t_scalar,
            encoder_attention_mask=None,
            return_all_hidden_states=False,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attn_mask,
        )

        # Neuron forward uses the projected temb directly
        neuron_out = neuron_model(sa_embs, cond, ref_temb, cross_mask)

    print(f"\n  ref_out:    {ref_out.shape}")
    print(f"  neuron_out: {neuron_out.shape}")
    assert ref_out.shape == neuron_out.shape, f"Shape mismatch: {ref_out.shape} vs {neuron_out.shape}"

    mean_diff, cos_sim = _report("dit_correctness", ref_out, neuron_out)

    assert mean_diff <= 0.15, f"mean_diff {mean_diff:.5f} exceeds threshold 0.15"
    assert cos_sim >= 0.999, f"cosine_sim {cos_sim:.6f} below threshold 0.999"
    print("  PASS ✓")


if __name__ == '__main__':
    import json
    print("Loading checkpoint...")
    sd = {}
    for f in sorted(glob.glob(os.path.join(MODEL_PATH, '*.safetensors'))):
        sd.update(load_file(f))

    from gr00t.model.modules.dit import AlternateVLDiT
    with open(os.path.join(MODEL_PATH, 'config.json')) as f:
        cfg = json.load(f)
    dit_cfg = cfg['diffusion_model_cfg']
    ref = AlternateVLDiT(**dit_cfg, cross_attention_dim=cfg['backbone_embedding_dim'],
                         attend_text_every_n_blocks=cfg.get('attend_text_every_n_blocks', 2)
                         ).bfloat16().eval()
    prefix = 'action_head.model.'
    ref.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}, strict=False)

    neu = GR00TDiTModel().bfloat16().eval()
    neu.load_state_dict(load_dit_model_state_dict(sd), strict=True)

    test_dit_correctness(neu, ref)
    print("ALL PASS")
