"""
vlm_backbone_block.py — GR00TBackboneLLM + NeuronGR00TBackboneWrapper

16-layer Qwen3-VL-2B LLM backbone for GR00T feature extraction.
Compiled via ModelBuilder with TP=8.

ViT (vision encoder) runs on CPU and produces inputs_embeds [B, 256, 2048].
This compiled subgraph takes pre-embedded inputs and produces conditioning tokens.

Architecture:
  - 16 NeuronQwen3VLDecoderLayer (GQA: 16 Q-heads, 8 KV-heads, head_dim=128)
  - mRoPE via NeuronQwen3VLRotaryEmbedding (rotary_position_ids: [3, B, SEQ])
  - Final RMSNorm
  - Output: [B, NUM_CONDITIONING_TOKENS, 2048] BF16

Forward inputs (static shapes, all CPU-precomputed):
  inputs_embeds:        [B, NUM_CONDITIONING_TOKENS, 2048]  BF16
  rotary_position_ids:  [3, B, NUM_CONDITIONING_TOKENS]     INT64 (3D mRoPE)

CRITICAL: GR00TBackboneLLM is NOT constructed in __init__. Constructed in
load_module() AFTER ModelBuilder has initialized parallel_state(tp_degree=8).

Weight mapping from GR00T checkpoint → NeuronGR00TBackboneWrapper:
  backbone.model.model.language_model.layers.N.self_attn.q_proj.weight
    → model.layers.N.self_attn.qkv_proj.q_proj.weight
  backbone.model.model.language_model.layers.N.self_attn.o_proj.weight
    → model.layers.N.self_attn.o_proj.o_proj.weight
  backbone.model.model.language_model.layers.N.self_attn.q_norm.weight
    → model.layers.N.self_attn.q_layernorm.weight
  backbone.model.model.language_model.layers.N.self_attn.k_norm.weight
    → model.layers.N.self_attn.k_layernorm.weight
  All other keys: strip prefix, identity.
  embed_tokens.weight → model.embed_tokens.weight (kept for state_dict completeness)
  norm.weight → model.norm.weight
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
SKILLS_ROOT = os.path.join(ROOT, 'skills', 'scripts')
sys.path.insert(0, ROOT)
sys.path.insert(0, SKILLS_ROOT)
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')

from config_constants import (
    BATCH_SIZE, NUM_CONDITIONING_TOKENS, LLM_HIDDEN_SIZE,
    LLM_NUM_LAYERS, LLM_NUM_ATTENTION_HEADS, LLM_NUM_KV_HEADS,
    LLM_HEAD_DIM, LLM_INTERMEDIATE_SIZE, LLM_VOCAB_SIZE,
    LLM_ROPE_THETA, TP_DEGREE_VLM,
    VIT_PATCH_SIZE, VIT_TEMPORAL_PATCH_SIZE, VIT_SPATIAL_MERGE_SIZE,
    VISION_TOKENS_PER_IMAGE, MODEL_PATH,
)

# ---------------------------------------------------------------------------
# NxDI imports
# ---------------------------------------------------------------------------
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
    NeuronQwen3VLDecoderLayer,
    get_rmsnorm_cls,
)

try:
    from neuronx_distributed.parallel_layers import parallel_state
    def _parallel_initialized():
        return parallel_state.model_parallel_is_initialized()
except ImportError:
    def _parallel_initialized():
        return False

try:
    from neuron_action_head_base import NeuronActionHeadBase, NeuronDenoisingConfig
except ImportError:
    NeuronActionHeadBase = nn.Module
    NeuronDenoisingConfig = object

# Token IDs for ViT preprocessing
IMAGE_TOKEN_ID  = 151655
VISION_START    = 151652
VISION_END      = 151653
PAD_TOKEN       = 0
DUMMY_TEXT_TOKEN = 1000
NPATCH = 256
N_VIS  = VISION_TOKENS_PER_IMAGE   # 64


def _make_backbone_inference_config(tp_degree: int = TP_DEGREE_VLM) -> InferenceConfig:
    """InferenceConfig for 16-layer Qwen3-VL-2B backbone."""
    return InferenceConfig(
        neuron_config=NeuronConfig(
            tp_degree=tp_degree,
            batch_size=BATCH_SIZE,
            seq_len=NUM_CONDITIONING_TOKENS,
            torch_dtype=torch.bfloat16,
            on_cpu=(tp_degree == 1),
        ),
        hidden_size=LLM_HIDDEN_SIZE,
        num_hidden_layers=LLM_NUM_LAYERS,
        num_attention_heads=LLM_NUM_ATTENTION_HEADS,
        num_key_value_heads=LLM_NUM_KV_HEADS,
        head_dim=LLM_HEAD_DIM,
        intermediate_size=LLM_INTERMEDIATE_SIZE,
        vocab_size=LLM_VOCAB_SIZE,
        pad_token_id=0,
        rms_norm_eps=1e-6,
        rope_theta=LLM_ROPE_THETA,
        hidden_act='silu',
        rope_scaling={'rope_type': 'default', 'mrope_section': [24, 20, 20]},
        max_position_embeddings=32768,
        attention_bias=False,
        num_cores_per_group=1,
    )


# ---------------------------------------------------------------------------
# GR00TBackboneLLM — pure nn.Module, constructed inside load_module()
# ---------------------------------------------------------------------------

class GR00TBackboneLLM(nn.Module):
    """
    16 NeuronQwen3VLDecoderLayer + final RMSNorm.

    Takes pre-embedded inputs (ViT+embed ran on CPU) and produces
    hidden states [B, SEQ, 2048] for use as DiT conditioning tokens.

    IMPORTANT: This class must be instantiated AFTER parallel_state is initialized.
    Instantiate inside NeuronGR00TBackboneWrapper.load_module() only.
    """

    def __init__(self, cfg: InferenceConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [NeuronQwen3VLDecoderLayer(cfg) for _ in range(LLM_NUM_LAYERS)]
        )
        self.norm = get_rmsnorm_cls()(LLM_HIDDEN_SIZE, eps=1e-6)

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # [B, SEQ, 2048] BF16
        pre_cos: torch.Tensor,        # [B, SEQ, 128]  BF16 — HF-compatible mRoPE cos
        pre_sin: torch.Tensor,        # [B, SEQ, 128]  BF16 — HF-compatible mRoPE sin
    ) -> torch.Tensor:
        hidden = inputs_embeds
        # Pass pre-computed HF cos/sin as cos_cache/sin_cache kwargs.
        # NeuronAttentionBase.forward skips self.rotary_emb when cos_cache is not None,
        # using the provided values directly. This ensures NxDI uses the same mRoPE as HF.
        # attention_mask=None → NxDI causal_mask=False = full bidirectional attention.
        for layer in self.layers:
            out = layer(
                hidden,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                rotary_position_ids=None,
                cos_cache=pre_cos,
                sin_cache=pre_sin,
            )
            hidden = out[0]

        return self.norm(hidden)  # [B, SEQ, 2048]


# ---------------------------------------------------------------------------
# Wrapper for ModelBuilder compilation (TP=8)
# ---------------------------------------------------------------------------

class NeuronGR00TBackboneWrapper(nn.Module):
    """
    ModelWrapper-compatible wrapper for GR00TBackboneLLM.
    Lazy model init in load_module() — parallel_state must be active.
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.model      = None
        self._preload_sd = None
        self.tag        = 'backbone'

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # [B, NUM_CONDITIONING_TOKENS, 2048] BF16
        pre_cos: torch.Tensor,        # [B, NUM_CONDITIONING_TOKENS, 128]  BF16
        pre_sin: torch.Tensor,        # [B, NUM_CONDITIONING_TOKENS, 128]  BF16
    ) -> torch.Tensor:
        if self.model is None:
            self.load_module()
        return self.model(inputs_embeds, pre_cos, pre_sin)

    def input_generator(self):
        B  = BATCH_SIZE
        S  = NUM_CONDITIONING_TOKENS
        H  = LLM_HIDDEN_SIZE
        Dh = LLM_HEAD_DIM
        return [(
            torch.zeros(B, S, H,  dtype=torch.bfloat16),  # inputs_embeds
            torch.zeros(B, S, Dh, dtype=torch.bfloat16),  # pre_cos
            torch.zeros(B, S, Dh, dtype=torch.bfloat16),  # pre_sin
        )]

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if self.model is None:
            self._preload_sd = state_dict
            return nn.Module.load_state_dict(self, {}, strict=False)
        return nn.Module.load_state_dict(self, state_dict, strict=strict, **kwargs)

    def load_module(self):
        """
        ModelBuilder calls this AFTER parallel_state(tp_degree=8) is active.
        On CPU: sets NXD_CPU_MODE=1 so LlamaRMSNorm is used.
        """
        on_trainium = _parallel_initialized()
        tp = TP_DEGREE_VLM if on_trainium else 1

        prev_cpu_mode = os.environ.get('NXD_CPU_MODE', '0')
        if not on_trainium:
            os.environ['NXD_CPU_MODE'] = '1'

        cfg = _make_backbone_inference_config(tp_degree=tp)
        self.model = GR00TBackboneLLM(cfg)

        if not on_trainium:
            os.environ['NXD_CPU_MODE'] = prev_cpu_mode

        if self._preload_sd is not None and not on_trainium:
            model_pfx = 'model.'
            stripped = {k[len(model_pfx):] if k.startswith(model_pfx) else k: v
                        for k, v in self._preload_sd.items()}
            missing, unexpected = self.model.load_state_dict(stripped, strict=False)
            non_buf = [k for k in missing if 'inv_freq' not in k]
            if non_buf:
                raise RuntimeError(f'Missing backbone keys: {non_buf[:5]}')

        self.model = self.model.bfloat16().eval()

    def is_neuron(self):
        return self.model is not None and isinstance(self.model, torch.jit.ScriptModule)

    def get(self, bucket_rank=0, **kwargs):
        return self, {}


# ---------------------------------------------------------------------------
# Backbone lifecycle head
# ---------------------------------------------------------------------------

def _make_backbone_compile_config():
    return NeuronDenoisingConfig(
        batch_size=BATCH_SIZE,
        tp_degree=TP_DEGREE_VLM,
        action_chunk_size=NUM_CONDITIONING_TOKENS,
        action_dim=LLM_HIDDEN_SIZE,
        num_conditioning_tokens=NUM_CONDITIONING_TOKENS,
        conditioning_hidden_size=LLM_HIDDEN_SIZE,
        timestep_embed_dim=LLM_HIDDEN_SIZE,
    )


class NeuronGR00TBackboneHead(NeuronActionHeadBase):
    """Lifecycle owner for backbone compilation via ModelBuilder TP=8."""

    def __init__(self, model_path, config, state_dict=None):
        super().__init__(model_path, config)
        self._preload_sd = state_dict

    def _build_denoising_wrapper(self):
        wrapper = NeuronGR00TBackboneWrapper()
        wrapper._preload_sd = self._preload_sd
        return wrapper

    def _build_attention_mask(self):
        return None

    def checkpoint_loader_fn(self, mmap=False):
        return self._preload_sd if self._preload_sd else {}

    def get_compiler_args(self):
        return (
            "--auto-cast=none "
            "--model-type=transformer "
            "--tensorizer-options='"
            "--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=1'"
        )

    def get_conditioning_contract(self):
        from neuron_action_head_base import ConditioningContract
        return ConditioningContract(
            num_conditioning_tokens=NUM_CONDITIONING_TOKENS,
            conditioning_hidden_size=LLM_HIDDEN_SIZE,
        )

    def _get_timestep_sequence(self, num_steps):
        return []

    def _embed_timestep(self, t):
        return torch.zeros(BATCH_SIZE, LLM_HIDDEN_SIZE, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Weight mapping: GR00T checkpoint → NeuronGR00TBackboneWrapper
# ---------------------------------------------------------------------------

def get_backbone_key_mapping() -> dict:
    """
    Map backbone.model.model.language_model.* → model.layers.N.* keys.

    Key renames matching NeuronQwen3VLDecoderLayer attribute names:
      self_attn.q_proj.* → self_attn.qkv_proj.q_proj.*
      self_attn.k_proj.* → self_attn.qkv_proj.k_proj.*
      self_attn.v_proj.* → self_attn.qkv_proj.v_proj.*
      self_attn.o_proj.* → self_attn.o_proj.o_proj.*
      self_attn.q_norm.* → self_attn.q_layernorm.*
      self_attn.k_norm.* → self_attn.k_layernorm.*
    """
    src = 'backbone.model.model.language_model.'
    mapping = {}
    for i in range(LLM_NUM_LAYERS):
        for hf_sub, nxd_sub in [
            (f'layers.{i}.self_attn.q_proj.weight', f'layers.{i}.self_attn.qkv_proj.q_proj.weight'),
            (f'layers.{i}.self_attn.k_proj.weight', f'layers.{i}.self_attn.qkv_proj.k_proj.weight'),
            (f'layers.{i}.self_attn.v_proj.weight', f'layers.{i}.self_attn.qkv_proj.v_proj.weight'),
            (f'layers.{i}.self_attn.o_proj.weight', f'layers.{i}.self_attn.o_proj.o_proj.weight'),
            (f'layers.{i}.self_attn.q_norm.weight', f'layers.{i}.self_attn.q_layernorm.weight'),
            (f'layers.{i}.self_attn.k_norm.weight', f'layers.{i}.self_attn.k_layernorm.weight'),
            (f'layers.{i}.mlp.gate_proj.weight', f'layers.{i}.mlp.gate_proj.weight'),
            (f'layers.{i}.mlp.up_proj.weight',   f'layers.{i}.mlp.up_proj.weight'),
            (f'layers.{i}.mlp.down_proj.weight',  f'layers.{i}.mlp.down_proj.weight'),
            (f'layers.{i}.input_layernorm.weight',          f'layers.{i}.input_layernorm.weight'),
            (f'layers.{i}.post_attention_layernorm.weight', f'layers.{i}.post_attention_layernorm.weight'),
        ]:
            mapping[src + hf_sub] = 'model.' + nxd_sub

    mapping[src + 'norm.weight'] = 'model.norm.weight'
    return mapping


def load_backbone_state_dict(hf_sd: dict) -> dict:
    mapping = get_backbone_key_mapping()
    out, missing = {}, []
    for hf_key, nxd_key in mapping.items():
        if hf_key in hf_sd:
            out[nxd_key] = hf_sd[hf_key]
        else:
            missing.append(hf_key)
    if missing:
        raise KeyError(f'Missing {len(missing)} backbone keys, e.g. {missing[:3]}')
    return out


def load_backbone_model_state_dict(hf_sd: dict) -> dict:
    return {k.removeprefix('model.'): v for k, v in load_backbone_state_dict(hf_sd).items()}


# ---------------------------------------------------------------------------
# CPU ViT preprocessing (runs on CPU, not compiled)
# ---------------------------------------------------------------------------

_FULL_MODEL_CACHE = {}


def _get_full_qwen_model(hf_sd=None):
    if 'model' not in _FULL_MODEL_CACHE:
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        from config_constants import LLM_TOTAL_LAYERS
        qwen_cfg = Qwen3VLConfig(
            text_config=dict(
                hidden_size=2048, num_hidden_layers=LLM_TOTAL_LAYERS,
                num_attention_heads=16, num_key_value_heads=8,
                head_dim=128, intermediate_size=6144,
                vocab_size=151936, max_position_embeddings=32768,
                rope_theta=LLM_ROPE_THETA, hidden_act='silu',
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
            image_token_id=IMAGE_TOKEN_ID,
        )
        qwen_cfg._attn_implementation = 'eager'
        full = Qwen3VLForConditionalGeneration(qwen_cfg).bfloat16().eval()
        if hf_sd is not None:
            pfx = 'backbone.model.'
            backbone_sd = {k[len(pfx):]: v for k, v in hf_sd.items() if k.startswith(pfx)}
            full.load_state_dict(backbone_sd, strict=False)
        _FULL_MODEL_CACHE['model'] = full
    return _FULL_MODEL_CACHE['model']


def run_vit_cpu(input_ids, attention_mask, pixel_values, image_grid_thw, hf_sd=None):
    """
    Run ViT + embed + scatter on CPU.
    Returns (inputs_embeds, pre_cos, pre_sin) for backbone NEFF.

    pre_cos/pre_sin use HF's Qwen3VLTextRotaryEmbedding (injected into NEFF via
    cos_cache/sin_cache kwargs, bypassing NxDI's NeuronQwen3VLRotaryEmbedding).
    """
    full = _get_full_qwen_model(hf_sd)
    with torch.no_grad():
        inputs_embeds = full.model.get_input_embeddings()(input_ids)
        image_embeds, _ = full.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.dtype)
        image_mask, _ = full.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds_cat
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)
        position_ids, _ = full.model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )
        # Compute HF-compatible cos/sin using HF's own rotary embedding
        hf_rope = full.model.language_model.rotary_emb
        pre_cos, pre_sin = hf_rope(inputs_embeds, position_ids)
    return inputs_embeds, pre_cos.to(inputs_embeds.dtype), pre_sin.to(inputs_embeds.dtype)


def make_backbone_inputs(B: int = BATCH_SIZE, n_text: int = 16) -> dict:
    SEQ = NUM_CONDITIONING_TOKENS
    input_ids = torch.full((B, SEQ), PAD_TOKEN, dtype=torch.long)
    for b in range(B):
        input_ids[b, 0] = VISION_START
        input_ids[b, 1:1+N_VIS] = IMAGE_TOKEN_ID
        input_ids[b, 1+N_VIS] = VISION_END
        input_ids[b, 2+N_VIS:2+N_VIS+n_text] = DUMMY_TEXT_TOKEN
    attn_mask = (input_ids != PAD_TOKEN).long()
    pixel_values = torch.randn(
        NPATCH, 3 * VIT_TEMPORAL_PATCH_SIZE * VIT_PATCH_SIZE * VIT_PATCH_SIZE,
        dtype=torch.bfloat16,
    )
    return {
        'input_ids': input_ids,
        'attention_mask': attn_mask,
        'pixel_values': pixel_values,
        'image_grid_thw': torch.tensor([[1, 16, 16]], dtype=torch.long),
    }
