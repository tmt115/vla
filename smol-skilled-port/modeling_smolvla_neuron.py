"""
modeling_smolvla_neuron.py

Phase 3: NxDI scaffolding for SmolVLA → AWS Trainium port.

Assembles the three Phase 2 compilation units into NxDI application heads:
  - NeuronSmolVLAVisionHead    (NeuronBaseForCausalLM): SigLIP + connector
  - NeuronSmolVLABackboneHead  (NeuronBaseForCausalLM): SmolLM2 prefix pass
  - NeuronSmolVLAActionHead    (NeuronActionHeadBase):  interleaved denoiser
  - NeuronSmolVLAPolicy: top-level orchestrator (build → load → compile → run)

convert_hf_to_neuron_state_dict is a placeholder for Phase 4.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import List, Type

sys.path.insert(0, '/home/ubuntu/smol-port-skilled')
sys.path.insert(0, '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages')
sys.path.insert(0, '/home/ubuntu/smol-port-skilled/skills/scripts')

try:
    from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
    from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
    _NEURON_AVAILABLE = True
except ImportError:
    _NEURON_AVAILABLE = False
    InferenceConfig = object
    NeuronConfig = object
    NeuronBaseModel = nn.Module
    NeuronBaseForCausalLM = nn.Module

from config_constants import (
    BATCH_SIZE,
    VISION_IMAGE_SIZE,
    HIDDEN_SIZE,
    PREFIX_SEQ_LEN,
    NUM_TEXT_TOKENS,
    NUM_STATE_TOKENS,
    VISION_TOKENS_PER_CAMERA,
    CONDITIONING_HIDDEN_SIZE,
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    NUM_CONDITIONING_TOKENS,
    NUM_DENOISING_STEPS,
    TIMESTEP_EMBED_DIM,
    MIN_PERIOD,
    MAX_PERIOD,
)
from vision_encoder_block import NeuronVisionEncoderBlock
from vlm_backbone_block import NeuronVLMBackboneBlock
from action_expert_block import (
    NeuronSmolVLAActionHead as _NeuronSmolVLAActionHead,
    NeuronSmolVLADenoisingWrapper,
    load_from_checkpoint as _load_expert_from_checkpoint,
    SmolVLAActionHeadConfig,
)


# ---------------------------------------------------------------------------
# 1. NeuronConfig subclass — per-subgraph TP degrees
# ---------------------------------------------------------------------------

class SmolVLANeuronConfig(NeuronConfig):
    """Extends NeuronConfig with per-subgraph tensor-parallel degrees for SmolVLA."""

    def __init__(self, **kwargs):
        # Pop custom kwargs before passing to NeuronConfig (it rejects unknown keys)
        self.tp_degree_vision = kwargs.pop('tp_degree_vision', 1)
        self.tp_degree_vlm    = kwargs.pop('tp_degree_vlm', 8)
        self.tp_degree_expert = kwargs.pop('tp_degree_expert', 2)
        if _NEURON_AVAILABLE:
            super().__init__(**kwargs)
        else:
            # Store remaining kwargs (e.g. tp_degree) on self
            for k, v in kwargs.items():
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# 2. InferenceConfig subclass — surfaces HF config fields
# ---------------------------------------------------------------------------

class SmolVLAInferenceConfig(InferenceConfig):
    """
    Surfaces SmolVLA HF config fields required at inference time.
    Binds SmolVLANeuronConfig as the Neuron config class.
    """

    def get_required_attributes(self) -> List[str]:
        return [
            # VLM (SmolLM2) fields
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "intermediate_size",
            "rms_norm_eps",
            "max_position_embeddings",
            "vocab_size",
            # Vision (SigLIP) fields — nested under vision_config in HF
            "vision_config",
            # Action head fields
            "action_chunk_size",
            "action_dim",
            "state_dim",
            "num_denoising_steps",
            "min_period",
            "max_period",
            "timestep_embed_dim",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type:
        return SmolVLANeuronConfig


# ---------------------------------------------------------------------------
# 3. NeuronBaseModel subclasses — compiled graph bodies
# ---------------------------------------------------------------------------

class NeuronSmolVLAVisionModel(NeuronBaseModel):
    """
    SigLIP ViT + pixel-shuffle connector.
    Input:  pixel_values [B, 3, 512, 512] BF16
    Output: vision_embs  [B,  64,    960] BF16
    """

    def setup_attr_for_model(self, config: InferenceConfig):
        neuron_cfg = config.neuron_config
        self.tp_degree   = getattr(neuron_cfg, 'tp_degree_vision', 1)
        self.hidden_size = HIDDEN_SIZE
        self.batch_size  = getattr(neuron_cfg, 'batch_size', BATCH_SIZE)

    def init_model(self, config: InferenceConfig):
        self.encoder = NeuronVisionEncoderBlock()

    def __init__(self, config=None):
        if _NEURON_AVAILABLE and config is not None:
            super().__init__(config)
        else:
            nn.Module.__init__(self)
            self.encoder = NeuronVisionEncoderBlock()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values)


class NeuronSmolVLABackboneModel(NeuronBaseModel):
    """
    SmolLM2 prefix pass (16 layers).
    Input:  prefix_embs [B, 113, 960], attn_mask [B, 113, 113] bool, pos_ids [B, 113] int64
    Output: conditioning [B, 113, 5120] BF16
    """

    def setup_attr_for_model(self, config: InferenceConfig):
        neuron_cfg = config.neuron_config
        self.tp_degree   = getattr(neuron_cfg, 'tp_degree_vlm', 8)
        self.hidden_size = HIDDEN_SIZE
        self.batch_size  = getattr(neuron_cfg, 'batch_size', BATCH_SIZE)

    def init_model(self, config: InferenceConfig):
        self.backbone = NeuronVLMBackboneBlock()

    def __init__(self, config=None):
        if _NEURON_AVAILABLE and config is not None:
            super().__init__(config)
        else:
            nn.Module.__init__(self)
            self.backbone = NeuronVLMBackboneBlock()

    def forward(
        self,
        prefix_embs: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone(prefix_embs, attn_mask, pos_ids)


# ---------------------------------------------------------------------------
# 4. Application heads — NeuronBaseForCausalLM wrapping each body
# ---------------------------------------------------------------------------

class NeuronSmolVLAVisionHead(NeuronBaseForCausalLM):
    """Application head for the vision encoder compilation unit."""

    _model_cls = NeuronSmolVLAVisionModel

    @classmethod
    def get_config_cls(cls):
        return SmolVLAInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        """
        Map HF checkpoint keys → NeuronSmolVLAVisionModel state dict keys.

        HF: model.vlm_with_expert.vlm.model.vision_model.{rest}
          → encoder.{rest}  (with encoder.layers.{i} → encoder.{i})

        HF: model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight
          → encoder.connector.proj.weight
        """
        VISION_HF  = "model.vlm_with_expert.vlm.model.vision_model."
        CONN_HF    = "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"
        CONN_NEU   = "encoder.connector.proj.weight"

        new_sd = {}
        for hf_key, tensor in state_dict.items():
            if hf_key == CONN_HF:
                new_sd[CONN_NEU] = tensor
            elif hf_key.startswith(VISION_HF):
                rest = hf_key[len(VISION_HF):]
                # encoder.layers.{i}.* → encoder.{i}.*
                if rest.startswith("encoder.layers."):
                    rest = rest.replace("encoder.layers.", "encoder.", 1)
                new_sd["encoder." + rest] = tensor
            # all other HF keys (lm_head, embed_tokens, text_model, lm_expert, action_*)
            # are silently dropped — they belong to other subgraphs
        return new_sd


class NeuronSmolVLABackboneHead(NeuronBaseForCausalLM):
    """Application head for the VLM backbone compilation unit."""

    _model_cls = NeuronSmolVLABackboneModel

    @classmethod
    def get_config_cls(cls):
        return SmolVLAInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        """
        Map HF checkpoint keys → NeuronSmolVLABackboneModel state dict keys.

        HF: model.vlm_with_expert.vlm.model.text_model.{rest}
          → backbone.{rest}

        Skips: embed_tokens.weight (not consumed by prefix-pass block),
               lm_head.weight, and all non-text-model keys.
        """
        TEXT_HF  = "model.vlm_with_expert.vlm.model.text_model."
        SKIP_SUFFIXES = {"embed_tokens.weight"}

        new_sd = {}
        for hf_key, tensor in state_dict.items():
            if hf_key.startswith(TEXT_HF):
                rest = hf_key[len(TEXT_HF):]
                if rest not in SKIP_SUFFIXES:
                    new_sd["backbone." + rest] = tensor
        return new_sd


# ---------------------------------------------------------------------------
# 5. Action expert head — re-export from action_expert_block for single-module imports
# ---------------------------------------------------------------------------

NeuronSmolVLAActionHead = _NeuronSmolVLAActionHead


# ---------------------------------------------------------------------------
# 6. Top-level policy orchestrator
# ---------------------------------------------------------------------------

class NeuronSmolVLAPolicy(nn.Module):
    """
    Top-level SmolVLA policy on AWS Trainium.

    Orchestrates:
      1. Vision encoder:  pixel_values → vision_embeddings [B, 64, 960]
      2. VLM backbone:    prefix_embs  → conditioning      [B, 113, 5120]
      3. Action expert:   10-step flow-matching denoising  → actions [B, 50, 32]

    Workflow (on Trainium):
        policy = NeuronSmolVLAPolicy()
        policy.load_weights(checkpoint_path)    # Phase 4: loads + converts HF weights
        policy.compile(compiled_dir)            # Phase 5: traces → NEFFs
        actions = policy.run(pixel_values, text_embs, state_embs)
    """

    def __init__(self, neuron_config=None):
        super().__init__()
        action_cfg = SmolVLAActionHeadConfig()

        self.vision_model   = NeuronSmolVLAVisionModel()
        self.backbone_model = NeuronSmolVLABackboneModel()
        self.action_head    = NeuronSmolVLAActionHead(action_cfg)

    def load_weights(self, checkpoint_path: str) -> None:
        """
        Load weights from the SmolVLA safetensors checkpoint.

        Vision + VLM: weight conversion implemented in Phase 4 via
        NeuronSmolVLAVisionHead.convert_hf_to_neuron_state_dict and
        NeuronSmolVLABackboneHead.convert_hf_to_neuron_state_dict.

        Action expert: uses the separate lm_expert weight path
        (model.vlm_with_expert.lm_expert.*) — NOT in the AutoModelForImageToText
        state dict. load_from_checkpoint handles this remapping directly.
        """
        if self.action_head.denoising_wrapper is None:
            raise RuntimeError(
                "Call compile() before load_weights() so denoising_wrapper is initialized."
            )
        _load_expert_from_checkpoint(self.action_head.denoising_wrapper, checkpoint_path)
        # Vision + VLM weight loading: implemented in Phase 4

    def compile(self, compiled_dir: str) -> None:
        """
        Compile all three subgraphs and save NEFFs to compiled_dir.

        - Action expert:  compile_denoiser() → compiled_dir/action_head/
        - Vision encoder: torch_neuronx.trace() on NeuronSmolVLAVisionHead  (Phase 5)
        - VLM backbone:   torch_neuronx.trace() on NeuronSmolVLABackboneHead (Phase 5)
        """
        os.makedirs(compiled_dir, exist_ok=True)
        action_dir = os.path.join(compiled_dir, "action_head")
        os.makedirs(action_dir, exist_ok=True)
        self.action_head.compile_denoiser(save_path=action_dir)

    def run(
        self,
        pixel_values: torch.Tensor,             # [B, 3, 512, 512] BF16
        text_embs: torch.Tensor,                # [B, 48, 960]
        state_embs: torch.Tensor,               # [B, 1, 960]
        num_denoising_steps: int = NUM_DENOISING_STEPS,
    ) -> torch.Tensor:                          # [B, 50, 32] BF16
        """Full inference: pixel values + text + state → predicted action chunk."""
        assert self.action_head.denoising_wrapper is not None, (
            "Call compile() before run()."
        )
        B = pixel_values.shape[0]

        # Step 1: vision encoder
        with torch.no_grad():
            vision_embs = self.vision_model(pixel_values)   # [B, 64, 960]

        # Step 2: assemble prefix [text | vision | state] → [B, 113, 960]
        prefix = torch.cat([
            text_embs.to(vision_embs.dtype),
            vision_embs,
            state_embs.to(vision_embs.dtype),
        ], dim=1)

        # Step 3: VLM backbone prefix pass
        attn_mask = torch.ones(B, PREFIX_SEQ_LEN, PREFIX_SEQ_LEN, dtype=torch.bool)
        pos_ids   = torch.arange(PREFIX_SEQ_LEN).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            conditioning = self.backbone_model(prefix, attn_mask, pos_ids).to(torch.bfloat16)

        # Step 4: action expert denoising loop
        return self.action_head.generate_actions(conditioning, num_denoising_steps)
