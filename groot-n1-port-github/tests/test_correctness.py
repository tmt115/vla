"""
tests/test_correctness.py — Phase 6 correctness validation

Compares compiled NEFF output against the original HuggingFace / Isaac-GR00T models.
This is the real correctness gate — NEFF vs HF model, not NEFF vs our re-implementation.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/groot-n1-port
    python -m pytest tests/test_correctness.py -v -s

Thresholds (bfloat16, single subgraph):
  Backbone LLM (16L):      mean_diff <= 0.10,  cos_sim >= 0.997
  VL self-attention (4L):  mean_diff <= 0.05,  cos_sim >= 0.999
  DiT action head (32L):   mean_diff <= 0.15,  cos_sim >= 0.999
"""
import sys, os, glob, json
sys.path.insert(0, '/home/ubuntu/groot-n1-port')
sys.path.insert(0, '/home/ubuntu/groot-port/skills/scripts')
sys.path.insert(0, '/home/ubuntu/Isaac-GR00T')
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

import torch
import pytest
import torch_neuronx  # must import before torch.jit.load

from safetensors.torch import load_file
from config_constants import *
from vlm_backbone_block import run_vit_cpu, make_backbone_inputs

MODEL_PATH_LOCAL = (
    '/home/ubuntu/.cache/huggingface/hub/'
    'models--nvidia--GR00T-N1.7-3B/snapshots/'
    '2fc962b973bccdd5d8ce4f67cc63b264d6886495'
)
# TP=1 verification NEFFs (compiled with torch_neuronx.trace(), weights baked in)
COMPILED_DIR = '/home/ubuntu/groot-n1-port/compiled_tp1'


def _report(name, cpu_out, neff_out, max_mean_diff, min_cos_sim):
    diff = (cpu_out.float() - neff_out.float()).abs()
    mean_diff = diff.mean().item()
    max_diff  = diff.max().item()
    a = cpu_out.float().flatten()
    b = neff_out.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()
    print(f'\n  {name}: mean_diff={mean_diff:.5f}  max_diff={max_diff:.4f}  cos_sim={cos_sim:.6f}')
    assert mean_diff <= max_mean_diff, f'{name}: mean_diff {mean_diff:.5f} > {max_mean_diff}'
    assert cos_sim >= min_cos_sim,     f'{name}: cos_sim {cos_sim:.6f} < {min_cos_sim}'
    return mean_diff, cos_sim


@pytest.fixture(scope='module')
def hf_sd():
    sd = {}
    for f in sorted(glob.glob(os.path.join(MODEL_PATH_LOCAL, '*.safetensors'))):
        sd.update(load_file(f))
    return sd


# ---------------------------------------------------------------------------
# Test 1: Backbone LLM — NEFF vs Qwen3VLForConditionalGeneration (HF)
# ---------------------------------------------------------------------------

def test_backbone_correctness(hf_sd):
    neff_path = os.path.join(COMPILED_DIR, 'backbone', 'model.pt')
    assert os.path.exists(neff_path), f'NEFF not found: {neff_path}'
    print('\n── Backbone LLM: NEFF vs HF Qwen3VLForConditionalGeneration ──')

    # HF reference: full Qwen3VL model on CPU (the exact model the backbone is compiled from)
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
    qwen_cfg = Qwen3VLConfig(
        text_config=dict(
            hidden_size=2048, num_hidden_layers=28,
            num_attention_heads=16, num_key_value_heads=8, head_dim=128,
            intermediate_size=6144, vocab_size=151936,
            max_position_embeddings=32768, rope_theta=5e6, hidden_act='silu',
            rope_scaling={'rope_type': 'default', 'mrope_section': [24, 20, 20]},
        ),
        vision_config=dict(
            depth=24, hidden_size=1024, hidden_act='quick_gelu',
            intermediate_size=4096, num_heads=16, in_channels=3,
            patch_size=VIT_PATCH_SIZE, spatial_merge_size=VIT_SPATIAL_MERGE_SIZE,
            temporal_patch_size=VIT_TEMPORAL_PATCH_SIZE,
            out_hidden_size=2048, num_position_embeddings=2304,
            deepstack_visual_indexes=[5, 11, 17],
        ),
        image_token_id=151655,
    )
    qwen_cfg._attn_implementation = 'eager'
    ref = Qwen3VLForConditionalGeneration(qwen_cfg).bfloat16().eval()
    # Truncate to select_layer=16
    while len(ref.model.language_model.layers) > LLM_NUM_LAYERS:
        ref.model.language_model.layers.pop(-1)
    pfx = 'backbone.model.'
    ref.load_state_dict({k[len(pfx):]: v for k, v in hf_sd.items() if k.startswith(pfx)}, strict=False)

    # NEFF
    neff = torch.jit.load(neff_path)

    # Inputs: ViT on CPU → inputs_embeds + position_ids (shared between ref and NEFF)
    inp = make_backbone_inputs()
    inputs_embeds, position_ids = run_vit_cpu(**inp, hf_sd=hf_sd)

    with torch.no_grad():
        # HF reference: run LLM portion with pre-computed inputs_embeds
        ref_out = ref.model.language_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids,
            output_hidden_states=True, use_cache=False,
        ).hidden_states[-1]
        neff_out = neff(inputs_embeds, position_ids)

    print(f'  shapes: ref={ref_out.shape}  neff={neff_out.shape}')
    assert ref_out.shape == neff_out.shape
    _report('backbone_llm', ref_out, neff_out.to(ref_out.dtype), 0.10, 0.997)
    print('  PASS ✓')


# ---------------------------------------------------------------------------
# Test 2: VL self-attention — NEFF vs SelfAttentionTransformer (Isaac-GR00T)
# ---------------------------------------------------------------------------

def test_vl_self_attn_correctness(hf_sd):
    neff_path = os.path.join(COMPILED_DIR, 'vl_self_attn', 'model.pt')
    assert os.path.exists(neff_path), f'NEFF not found: {neff_path}'
    print('\n── VL self-attention: NEFF vs SelfAttentionTransformer (HF) ──')

    from gr00t.model.modules.dit import SelfAttentionTransformer
    with open(os.path.join(MODEL_PATH_LOCAL, 'config.json')) as f:
        groot_cfg = json.load(f)

    ref = SelfAttentionTransformer(**groot_cfg['vl_self_attention_cfg']).bfloat16().eval()
    pfx = 'action_head.vl_self_attention.'
    ref.load_state_dict(
        {k[len(pfx):]: v for k, v in hf_sd.items() if k.startswith(pfx)},
        strict=False,
    )

    neff = torch.jit.load(neff_path)

    torch.manual_seed(42)
    x = torch.randn(BATCH_SIZE, NUM_CONDITIONING_TOKENS,
                    VL_SELF_ATTN_HIDDEN_SIZE, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_out  = ref(x)
        neff_out = neff(x)

    print(f'  shapes: ref={ref_out.shape}  neff={neff_out.shape}')
    assert ref_out.shape == neff_out.shape
    _report('vl_self_attn', ref_out, neff_out.to(ref_out.dtype), 0.05, 0.999)
    print('  PASS ✓')


# ---------------------------------------------------------------------------
# Test 3: DiT — NEFF vs AlternateVLDiT (Isaac-GR00T)
# ---------------------------------------------------------------------------

def test_dit_correctness(hf_sd):
    neff_path = os.path.join(COMPILED_DIR, 'dit', 'model.pt')
    assert os.path.exists(neff_path), f'NEFF not found: {neff_path}'
    print('\n── DiT action head: NEFF vs AlternateVLDiT (HF) ──')

    from gr00t.model.modules.dit import AlternateVLDiT
    with open(os.path.join(MODEL_PATH_LOCAL, 'config.json')) as f:
        groot_cfg = json.load(f)

    ref = AlternateVLDiT(
        **groot_cfg['diffusion_model_cfg'],
        cross_attention_dim=groot_cfg['backbone_embedding_dim'],
        attend_text_every_n_blocks=groot_cfg.get('attend_text_every_n_blocks', 2),
    ).bfloat16().eval()
    pfx = 'action_head.model.'
    ref.load_state_dict(
        {k[len(pfx):]: v for k, v in hf_sd.items() if k.startswith(pfx)},
        strict=False,
    )

    neff = torch.jit.load(neff_path)

    torch.manual_seed(7)
    B = BATCH_SIZE
    sa_embs    = torch.randn(B, DIT_INPUT_SEQ_LEN,      DIT_HIDDEN_SIZE,          dtype=torch.bfloat16)
    cond       = torch.randn(B, NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE, dtype=torch.bfloat16)
    # Get temb from reference TimestepEncoder
    t_bucket = torch.tensor([500], dtype=torch.long)
    ref_temb = ref.timestep_encoder(t_bucket).bfloat16()
    cross_mask = torch.ones(B, 1, DIT_INPUT_SEQ_LEN, NUM_CONDITIONING_TOKENS, dtype=torch.int32)

    # Reference: AlternateVLDiT expects timestep as scalar bucket + image_mask
    image_mask    = torch.zeros(B, NUM_CONDITIONING_TOKENS, dtype=torch.bool)
    backbone_mask = torch.ones(B, NUM_CONDITIONING_TOKENS, dtype=torch.bool)
    with torch.no_grad():
        ref_out = ref(
            hidden_states=sa_embs,
            encoder_hidden_states=cond,
            timestep=t_bucket,
            image_mask=image_mask,
            backbone_attention_mask=backbone_mask,
        )
        # NEFF takes projected temb directly (TimestepEncoder runs on CPU in inference)
        neff_out = neff(sa_embs, cond, ref_temb, cross_mask)

    print(f'  shapes: ref={ref_out.shape}  neff={neff_out.shape}')
    assert ref_out.shape == neff_out.shape
    _report('dit_action_head', ref_out, neff_out.to(ref_out.dtype), 0.15, 0.999)
    print('  PASS ✓')


if __name__ == '__main__':
    print('Loading checkpoint...')
    sd = {}
    for f in sorted(glob.glob(os.path.join(MODEL_PATH_LOCAL, '*.safetensors'))):
        sd.update(load_file(f))

    test_backbone_correctness(sd)
    test_vl_self_attn_correctness(sd)
    test_dit_correctness(sd)

    print('\n' + '='*60)
    print('ALL CORRECTNESS CHECKS PASSED')
    print('='*60)
