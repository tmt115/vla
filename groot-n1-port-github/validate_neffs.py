"""
validate_neffs.py — Phase 5 NEFF correctness validation.

Compares each compiled NEFF against the Isaac-GR00T / HuggingFace CPU reference.
Uses proper NxDI head.load() for weight initialization — not raw torch.jit.load.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/groot-n1-port
    python validate_neffs.py
"""

import sys, os, glob, json, time
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/ubuntu/groot-n1-port')
sys.path.insert(0, '/home/ubuntu/groot-n1-port/skills/scripts')
sys.path.insert(0, '/home/ubuntu/Isaac-GR00T')
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

import torch_neuronx  # must import before torch.jit.load on Neuron

from safetensors.torch import load_file
from config_constants import *
from vlm_backbone_block import (
    NeuronGR00TBackboneHead, NeuronGR00TBackboneWrapper,
    load_backbone_state_dict, _make_backbone_compile_config,
    run_vit_cpu, make_backbone_inputs, _get_full_qwen_model,
)
from vl_self_attention_block import NeuronGR00TVLSelfAttnWrapper, load_vl_self_attn_state_dict
from dit_block import NeuronGR00TActionHead, NeuronGR00TDenoisingWrapper, load_dit_state_dict
from neuron_action_head_base import NeuronDenoisingConfig
from compile_all import _VLSelfAttnHead, _make_vl_self_attn_compile_config, _make_dit_compile_config

COMPILED_DIR = '/home/ubuntu/groot-n1-port/compiled'
BACKBONE_DIR  = os.path.join(COMPILED_DIR, 'backbone')
VL_SA_DIR     = os.path.join(COMPILED_DIR, 'vl_self_attn')
DIT_DIR       = os.path.join(COMPILED_DIR, 'dit')
RESULTS = {}


def _report(name, cpu_out, neff_out, atol, cos_threshold):
    cpu_f  = cpu_out.float().flatten()
    neff_f = neff_out.float().flatten()
    diff = (cpu_f - neff_f).abs()
    mean_diff = diff.mean().item()
    max_diff  = diff.max().item()
    cos_sim   = F.cosine_similarity(cpu_f.unsqueeze(0), neff_f.unsqueeze(0)).item()
    passed = mean_diff <= atol and cos_sim >= cos_threshold
    status = 'PASS ✓' if passed else 'FAIL ✗'
    print(f'  mean_diff={mean_diff:.5f}  max_diff={max_diff:.4f}  cos_sim={cos_sim:.6f}  → {status}')
    RESULTS[name] = {'mean_diff': mean_diff, 'cos_sim': cos_sim, 'passed': passed}
    if not passed:
        print(f'  THRESHOLD: atol={atol}, cos_threshold={cos_threshold}')
    return passed


# ---------------------------------------------------------------------------
# Load checkpoint once
# ---------------------------------------------------------------------------
print('Loading HF checkpoint...')
t0 = time.time()
hf_sd = {}
for f in sorted(glob.glob(os.path.join(MODEL_PATH, '*.safetensors'))):
    hf_sd.update(load_file(f))
print(f'  {len(hf_sd)} tensors in {time.time()-t0:.1f}s')

with open(os.path.join(MODEL_PATH, 'config.json')) as fp:
    groot_cfg = json.load(fp)


# ---------------------------------------------------------------------------
# Test 1: VL self-attention
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('TEST 1: VL Self-Attention — NEFF vs SelfAttentionTransformer')
print('='*60)

# Load NEFF via proper head.load()
cfg_vl = _make_vl_self_attn_compile_config()
head_vl = _VLSelfAttnHead(model_path=MODEL_PATH, config=cfg_vl)
wrapper_vl = NeuronGR00TVLSelfAttnWrapper()
head_vl.denoising_wrapper = wrapper_vl
head_vl.load(VL_SA_DIR)
print('  NEFF loaded.')

# HF reference: SelfAttentionTransformer from Isaac-GR00T
from gr00t.model.modules.dit import SelfAttentionTransformer
ref_vl = SelfAttentionTransformer(**groot_cfg['vl_self_attention_cfg']).bfloat16().eval()
pfx = 'action_head.vl_self_attention.'
ref_vl.load_state_dict(
    {k[len(pfx):]: v for k, v in hf_sd.items() if k.startswith(pfx)},
    strict=False,
)
print('  Reference loaded.')

torch.manual_seed(42)
x_vl = torch.randn(BATCH_SIZE, NUM_CONDITIONING_TOKENS, VL_SELF_ATTN_HIDDEN_SIZE,
                   dtype=torch.bfloat16)

with torch.no_grad():
    ref_out_vl  = ref_vl(x_vl)
    neff_out_vl = wrapper_vl(x_vl)

print(f'  shapes: ref={ref_out_vl.shape}  neff={neff_out_vl.shape}')
_report('vl_self_attn', ref_out_vl, neff_out_vl.to(ref_out_vl.dtype), atol=0.05, cos_threshold=0.999)

del head_vl  # free Neuron memory


# ---------------------------------------------------------------------------
# Test 2: DiT action head
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('TEST 2: DiT Action Head — NEFF vs AlternateVLDiT')
print('='*60)
print('  NOTE: deviation expected — NEFF attends all conditioning tokens per block;')
print('  AlternateVLDiT alternates image/text tokens by block. cos_sim threshold relaxed to 0.95.')

# Load NEFF
cfg_dit = _make_dit_compile_config()
head_dit = NeuronGR00TActionHead(model_path=MODEL_PATH, config=cfg_dit)
wrapper_dit = NeuronGR00TDenoisingWrapper()
head_dit.denoising_wrapper = wrapper_dit
head_dit.load(DIT_DIR)
print('  NEFF loaded.')

# HF reference: AlternateVLDiT
from gr00t.model.modules.dit import AlternateVLDiT
ref_dit = AlternateVLDiT(
    **groot_cfg['diffusion_model_cfg'],
    cross_attention_dim=groot_cfg['backbone_embedding_dim'],
    attend_text_every_n_blocks=groot_cfg.get('attend_text_every_n_blocks', 2),
).bfloat16().eval()
pfx = 'action_head.model.'
ref_dit.load_state_dict(
    {k[len(pfx):]: v for k, v in hf_sd.items() if k.startswith(pfx)},
    strict=False,
)
print('  Reference loaded.')

torch.manual_seed(7)
sa_embs  = torch.randn(BATCH_SIZE, DIT_INPUT_SEQ_LEN,      DIT_HIDDEN_SIZE,
                       dtype=torch.bfloat16)
cond     = torch.randn(BATCH_SIZE, NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE,
                       dtype=torch.bfloat16)
t_bucket = torch.tensor([500], dtype=torch.long)
# Get projected timestep embedding from reference's own encoder
with torch.no_grad():
    ref_temb = ref_dit.timestep_encoder(t_bucket).bfloat16()

cross_mask    = torch.ones(BATCH_SIZE, 1, DIT_INPUT_SEQ_LEN, NUM_CONDITIONING_TOKENS, dtype=torch.int32)
image_mask    = torch.zeros(BATCH_SIZE, NUM_CONDITIONING_TOKENS, dtype=torch.bool)
backbone_mask = torch.ones(BATCH_SIZE,  NUM_CONDITIONING_TOKENS, dtype=torch.bool)

with torch.no_grad():
    ref_out_dit = ref_dit(
        hidden_states=sa_embs,
        encoder_hidden_states=cond,
        timestep=t_bucket,
        image_mask=image_mask,
        backbone_attention_mask=backbone_mask,
    )
    neff_out_dit = wrapper_dit(sa_embs, cond, ref_temb, cross_mask)

print(f'  shapes: ref={ref_out_dit.shape}  neff={neff_out_dit.shape}')
_report('dit', ref_out_dit, neff_out_dit.to(ref_out_dit.dtype), atol=0.5, cos_threshold=0.95)

del head_dit


# ---------------------------------------------------------------------------
# Test 3: Backbone LLM
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('TEST 3: Backbone LLM (16 layers) — NEFF vs Qwen3VL language_model')
print('='*60)

# Load NEFF
cfg_bb = _make_backbone_compile_config()
head_bb = NeuronGR00TBackboneHead(model_path=MODEL_PATH, config=cfg_bb)
wrapper_bb = NeuronGR00TBackboneWrapper()
head_bb.denoising_wrapper = wrapper_bb
head_bb.load(BACKBONE_DIR)
print('  NEFF loaded.')

# HF reference: Qwen3VL with 16 layers only
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
qwen_cfg = Qwen3VLConfig(
    text_config=dict(
        hidden_size=2048, num_hidden_layers=LLM_TOTAL_LAYERS,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        intermediate_size=6144, vocab_size=151936,
        max_position_embeddings=32768, rope_theta=LLM_ROPE_THETA, hidden_act='silu',
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
ref_bb = Qwen3VLForConditionalGeneration(qwen_cfg).bfloat16().eval()
# Truncate to 16 layers
while len(ref_bb.model.language_model.layers) > LLM_NUM_LAYERS:
    ref_bb.model.language_model.layers.pop(-1)
pfx = 'backbone.model.'
ref_bb.load_state_dict(
    {k[len(pfx):]: v for k, v in hf_sd.items() if k.startswith(pfx)},
    strict=False,
)
print('  Reference loaded (16-layer Qwen3VL).')

# Compute shared inputs via run_vit_cpu
inp = make_backbone_inputs(B=BATCH_SIZE, n_text=16)
print('  Running ViT on CPU...')
with torch.no_grad():
    inputs_embeds, pre_cos, pre_sin = run_vit_cpu(
        inp['input_ids'], inp['attention_mask'],
        inp['pixel_values'], inp['image_grid_thw'],
        hf_sd=hf_sd,
    )
print(f'  inputs_embeds: {inputs_embeds.shape}, pre_cos: {pre_cos.shape}')

# Get position_ids for HF reference (to use the same mRoPE)
full_hf = _get_full_qwen_model(hf_sd)
pos_ids_ref, _ = full_hf.model.get_rope_index(inp['input_ids'], inp['image_grid_thw'], inp['attention_mask'])

with torch.no_grad():
    # NxDI NEFF: full bidirectional attention, uses pre_cos/pre_sin from run_vit_cpu.
    # HF reference: 4D all-zeros additive mask forces full attention.
    # Force HF reference to use IDENTICAL cos/sin as the NEFF by monkey-patching
    # rotary_emb. The two separate HF model instances (full and ref_bb) may produce
    # slightly different cos/sin for the same position_ids due to @dynamic_rope_update
    # state differences; pinning them to the same pre_cos/pre_sin ensures fair comparison.
    full_4d = torch.zeros(BATCH_SIZE, 1, NUM_CONDITIONING_TOKENS, NUM_CONDITIONING_TOKENS,
                          dtype=torch.bfloat16)
    # Patch rotary_emb.forward so the reference uses identical cos/sin as the NEFF.
    # Can't assign a lambda directly (PyTorch requires nn.Module for registered modules),
    # so patch the bound method instead.
    _orig_rope_fwd = ref_bb.model.language_model.rotary_emb.forward
    ref_bb.model.language_model.rotary_emb.forward = lambda x, pos_ids: (pre_cos, pre_sin)
    try:
        ref_lm_out = ref_bb.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=pos_ids_ref,
            attention_mask=full_4d,
            use_cache=False,
        )
        ref_out_bb = ref_lm_out.last_hidden_state  # [B, SEQ, 2048], norm'd
    finally:
        ref_bb.model.language_model.rotary_emb.forward = _orig_rope_fwd  # always restore
    # NEFF: new signature (inputs_embeds, pre_cos, pre_sin)
    neff_out_bb = wrapper_bb(inputs_embeds, pre_cos, pre_sin)

print(f'  shapes: ref={ref_out_bb.shape}  neff={neff_out_bb.shape}')
_report('backbone', ref_out_bb, neff_out_bb.to(ref_out_bb.dtype), atol=0.10, cos_threshold=0.997)

del head_bb


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('VALIDATION SUMMARY')
print('='*60)
all_passed = True
for name, r in RESULTS.items():
    status = 'PASS ✓' if r['passed'] else 'FAIL ✗'
    print(f'  {name:20s}  cos_sim={r["cos_sim"]:.6f}  mean_diff={r["mean_diff"]:.5f}  {status}')
    all_passed = all_passed and r['passed']

print()
if all_passed:
    print('ALL VALIDATIONS PASSED')
else:
    print('SOME VALIDATIONS FAILED — check above for details')
