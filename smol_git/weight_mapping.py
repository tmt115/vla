"""
weight_mapping.py
=================
Convert a SmolVLA HuggingFace / lerobot checkpoint state dict into the
separate state dicts expected by each NxDI-compiled module.

Usage
-----
    import torch
    from weight_mapping import convert_hf_to_neuron_state_dicts

    hf_sd = torch.load("smolvla_checkpoint.pt", map_location="cpu")
    # hf_sd may be the full SmolVLAPolicy state dict (keys start with "model.")
    # or just the VLAFlowMatching state dict (keys start with "vlm_with_expert.")

    parts = convert_hf_to_neuron_state_dicts(hf_sd)
    # parts is a dict with keys:
    #   "vision_encoder"  ->  NeuronSmolVLMVisionEncoder.load_state_dict(...)
    #   "connector"       ->  NeuronSmolVLMConnector.load_state_dict(...)
    #   "embed_tokens"    ->  nn.Embedding(49280, 960).load_state_dict(...)
    #   "prefix"          ->  NeuronSmolVLAPrefix.load_state_dict(...)
    #   "denoise_step"    ->  NeuronSmolVLADenoiseStep.load_state_dict(...)

HF checkpoint key structure
-----------------------------
SmolVLAPolicy.state_dict() (with "model." prefix):

  # --- Vision encoder (Idefics3 SigLIP ViT-L/16, 27 layers) ---
  model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight
  model.vlm_with_expert.vlm.model.vision_model.embeddings.position_embedding.weight
  model.vlm_with_expert.vlm.model.vision_model.encoder.layers.N.self_attn.{q,k,v,out_proj}.{weight,bias}
  model.vlm_with_expert.vlm.model.vision_model.encoder.layers.N.{layer_norm1,layer_norm2}.{weight,bias}
  model.vlm_with_expert.vlm.model.vision_model.encoder.layers.N.mlp.{fc1,fc2}.{weight,bias}
  model.vlm_with_expert.vlm.model.vision_model.post_layernorm.{weight,bias}

  # --- Connector (pixel-shuffle + single linear) ---
  model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight

  # --- VLM text model (SmolLM2 Llama, 16 layers used) ---
  model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight
  model.vlm_with_expert.vlm.model.text_model.layers.N.self_attn.{q,k,v,o}_proj.weight
  model.vlm_with_expert.vlm.model.text_model.layers.N.mlp.{gate,up,down}_proj.weight
  model.vlm_with_expert.vlm.model.text_model.layers.N.{input,post_attention}_layernorm.weight
  model.vlm_with_expert.vlm.model.text_model.norm.weight

  # --- Action expert (Llama GQA, 16 layers) ---
  model.vlm_with_expert.lm_expert.layers.N.self_attn.{q,k,v,o}_proj.weight
  model.vlm_with_expert.lm_expert.layers.N.mlp.{gate,up,down}_proj.weight
  model.vlm_with_expert.lm_expert.layers.N.{input,post_attention}_layernorm.weight

  # --- Action projections ---
  model.state_proj.{weight,bias}
  model.action_in_proj.{weight,bias}
  model.action_out_proj.{weight,bias}
  model.action_time_mlp_in.{weight,bias}
  model.action_time_mlp_out.{weight,bias}

NxDI module key structure
--------------------------
NeuronSmolVLMVisionEncoder:
  embeddings.patch_embedding.weight          (Conv2d, no bias)
  embeddings.position_embedding.weight
  encoder.N.self_attn.{q,k,v,out_proj}.{weight,bias}
  encoder.N.{layer_norm1,layer_norm2}.{weight,bias}
  encoder.N.mlp.{fc1,fc2}.{weight,bias}
  post_layernorm.{weight,bias}

NeuronSmolVLMConnector:
  proj.weight

NeuronSmolVLAPrefix:
  layers.N.self_attn.{q,k,v,o}_proj.weight
  layers.N.mlp.{gate,up,down}_proj.weight
  layers.N.{input_layernorm,post_attention_layernorm}.weight

NeuronSmolVLADenoiseStep:
  projections.state_proj.{weight,bias}
  projections.action_in_proj.{weight,bias}
  projections.action_out_proj.{weight,bias}
  projections.action_time_mlp_in.{weight,bias}
  projections.action_time_mlp_out.{weight,bias}
  projections.sine_scale           (buffer — deterministic, SKIP)
  expert_layers.N.self_attn.{q,k,v,o}_proj.weight
  expert_layers.N.mlp.{gate,up,down}_proj.weight
  expert_layers.N.{input_layernorm,post_attention_layernorm}.weight
"""

from __future__ import annotations
from typing import Dict
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Return a new dict with `prefix` removed from every key."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def _normalize(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Accept either SmolVLAPolicy state dict (keys start with "model.")
    or VLAFlowMatching state dict (keys start with "vlm_with_expert." / "state_proj." etc.).
    Returns a VLAFlowMatching-rooted dict.
    """
    # Check if the majority of keys carry a "model." prefix
    n_model = sum(1 for k in state_dict if k.startswith("model."))
    if n_model > len(state_dict) // 2:
        return _strip_prefix(state_dict, "model.")
    return dict(state_dict)


# ---------------------------------------------------------------------------
# Sub-dict builders
# ---------------------------------------------------------------------------

def _build_vision_encoder_sd(
    sd: Dict[str, torch.Tensor],
    num_layers: int = 27,
) -> Dict[str, torch.Tensor]:
    """
    Map SmolVLM vision_model keys → NeuronSmolVLMVisionEncoder keys.

    HF prefix: vlm_with_expert.vlm.model.vision_model.
    Neuron prefix: (none)
    """
    hf_pre = "vlm_with_expert.vlm.model.vision_model."
    out: Dict[str, torch.Tensor] = {}

    def _get(hf_key: str) -> torch.Tensor | None:
        full = hf_pre + hf_key
        return sd.get(full)

    def _put(neuron_key: str, hf_key: str):
        t = _get(hf_key)
        if t is not None:
            out[neuron_key] = t

    # Embeddings
    _put("embeddings.patch_embedding.weight", "embeddings.patch_embedding.weight")
    _put("embeddings.position_embedding.weight", "embeddings.position_embedding.weight")

    # Encoder layers
    for i in range(num_layers):
        hpfx = f"encoder.layers.{i}"
        npfx = f"encoder.{i}"

        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            for param in ("weight", "bias"):
                _put(f"{npfx}.self_attn.{proj}.{param}",
                     f"{hpfx}.self_attn.{proj}.{param}")

        for ln in ("layer_norm1", "layer_norm2"):
            for param in ("weight", "bias"):
                _put(f"{npfx}.{ln}.{param}", f"{hpfx}.{ln}.{param}")

        for fc in ("fc1", "fc2"):
            for param in ("weight", "bias"):
                _put(f"{npfx}.mlp.{fc}.{param}", f"{hpfx}.mlp.{fc}.{param}")

    # Post layer norm
    _put("post_layernorm.weight", "post_layernorm.weight")
    _put("post_layernorm.bias",   "post_layernorm.bias")

    return out


def _build_connector_sd(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map SmolVLM connector keys → NeuronSmolVLMConnector keys.

    HF:     vlm_with_expert.vlm.model.connector.modality_projection.proj.weight
    Neuron: proj.weight
    """
    hf_key = "vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"
    out: Dict[str, torch.Tensor] = {}
    if hf_key in sd:
        out["proj.weight"] = sd[hf_key]
    return out


def _build_embed_tokens_sd(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract the VLM text token embedding table.

    HF:     vlm_with_expert.vlm.model.text_model.embed_tokens.weight  [49280, 960]
    Neuron: weight  (load into nn.Embedding)
    """
    hf_key = "vlm_with_expert.vlm.model.text_model.embed_tokens.weight"
    out: Dict[str, torch.Tensor] = {}
    if hf_key in sd:
        out["weight"] = sd[hf_key]
    return out


def _build_prefix_sd(
    sd: Dict[str, torch.Tensor],
    num_layers: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Map VLM text_model layer keys → NeuronSmolVLAPrefix keys.

    HF prefix:     vlm_with_expert.vlm.model.text_model.layers.N.
    Neuron prefix: layers.N.
    """
    hf_pre = "vlm_with_expert.vlm.model.text_model.layers."
    out: Dict[str, torch.Tensor] = {}

    for i in range(num_layers):
        hp = f"{hf_pre}{i}."
        np_ = f"layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            src = f"{hp}self_attn.{proj}.weight"
            if src in sd:
                out[f"{np_}self_attn.{proj}.weight"] = sd[src]

        for proj in ("gate_proj", "up_proj", "down_proj"):
            src = f"{hp}mlp.{proj}.weight"
            if src in sd:
                out[f"{np_}mlp.{proj}.weight"] = sd[src]

        for ln in ("input_layernorm", "post_attention_layernorm"):
            src = f"{hp}{ln}.weight"
            if src in sd:
                out[f"{np_}{ln}.weight"] = sd[src]

    return out


def _build_denoise_step_sd(
    sd: Dict[str, torch.Tensor],
    num_expert_layers: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Map lm_expert layer keys + action projection keys → NeuronSmolVLADenoiseStep keys.

    Expert HF prefix:  vlm_with_expert.lm_expert.layers.N.
    Neuron prefix:     expert_layers.N.

    Action proj HF:    state_proj.* / action_in_proj.* / ...
    Neuron prefix:     projections.*

    Note: projections.sine_scale is a deterministic buffer and is NOT mapped
    (it is recomputed from scratch in NeuronSmolVLAActionProjections.__init__).
    """
    out: Dict[str, torch.Tensor] = {}

    # --- Expert layers ---
    exp_pre = "vlm_with_expert.lm_expert.layers."
    for i in range(num_expert_layers):
        hp = f"{exp_pre}{i}."
        np_ = f"expert_layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            src = f"{hp}self_attn.{proj}.weight"
            if src in sd:
                out[f"{np_}self_attn.{proj}.weight"] = sd[src]

        for proj in ("gate_proj", "up_proj", "down_proj"):
            src = f"{hp}mlp.{proj}.weight"
            if src in sd:
                out[f"{np_}mlp.{proj}.weight"] = sd[src]

        for ln in ("input_layernorm", "post_attention_layernorm"):
            src = f"{hp}{ln}.weight"
            if src in sd:
                out[f"{np_}{ln}.weight"] = sd[src]

    # --- Action projection layers (all have bias) ---
    proj_pairs = [
        ("state_proj",         "projections.state_proj"),
        ("action_in_proj",     "projections.action_in_proj"),
        ("action_out_proj",    "projections.action_out_proj"),
        ("action_time_mlp_in", "projections.action_time_mlp_in"),
        ("action_time_mlp_out","projections.action_time_mlp_out"),
    ]
    for hf_name, neuron_name in proj_pairs:
        for param in ("weight", "bias"):
            src = f"{hf_name}.{param}"
            if src in sd:
                out[f"{neuron_name}.{param}"] = sd[src]

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_hf_to_neuron_state_dicts(
    hf_state_dict: Dict[str, torch.Tensor],
    *,
    num_vlm_layers: int = 16,
    num_expert_layers: int = 16,
    num_vision_layers: int = 27,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Convert a SmolVLA HuggingFace checkpoint into per-module state dicts.

    Parameters
    ----------
    hf_state_dict:
        Full SmolVLAPolicy state dict (keys may start with "model.")
        OR VLAFlowMatching state dict (no "model." prefix).
    num_vlm_layers:
        Number of VLM text-model layers to map (default 16).
    num_expert_layers:
        Number of expert Llama layers to map (default 16).
    num_vision_layers:
        Number of vision encoder layers to map (default 27).

    Returns
    -------
    dict with keys:
        "vision_encoder"  : state dict for NeuronSmolVLMVisionEncoder
        "connector"       : state dict for NeuronSmolVLMConnector
        "embed_tokens"    : state dict for nn.Embedding (vocab → VLM hidden)
        "prefix"          : state dict for NeuronSmolVLAPrefix
        "denoise_step"    : state dict for NeuronSmolVLADenoiseStep
    """
    sd = _normalize(hf_state_dict)

    return {
        "vision_encoder": _build_vision_encoder_sd(sd, num_layers=num_vision_layers),
        "connector":      _build_connector_sd(sd),
        "embed_tokens":   _build_embed_tokens_sd(sd),
        "prefix":         _build_prefix_sd(sd, num_layers=num_vlm_layers),
        "denoise_step":   _build_denoise_step_sd(sd, num_expert_layers=num_expert_layers),
    }


def verify_coverage(
    hf_state_dict: Dict[str, torch.Tensor],
    parts: Dict[str, Dict[str, torch.Tensor]],
    *,
    verbose: bool = True,
) -> bool:
    """
    Check that every weight-bearing HF key is accounted for in some output dict.

    Skips non-weight keys (lm_head, text_model.norm, lm_expert.norm, etc.)
    that are not used by the compiled inference subgraphs.

    Returns True if no unexpected keys are missing.
    """
    sd = _normalize(hf_state_dict)

    # Keys that are intentionally excluded from inference subgraphs:
    # - lm_head (logits not needed at inference)
    # - text_model.norm (not used in prefix-KV-only pass)
    # - lm_expert.norm (not used; output goes straight to action_out_proj)
    # - lm_expert.embed_tokens (None in SmolVLMWithExpertModel)
    skip_patterns = (
        "vlm_with_expert.vlm.lm_head.",
        "vlm_with_expert.vlm.model.text_model.norm.",
        "vlm_with_expert.lm_expert.norm.",
        "vlm_with_expert.lm_expert.embed_tokens.",
    )

    all_mapped = set()
    for sub in parts.values():
        all_mapped.update(sub.keys())

    missing = []
    for hf_key in sd:
        if any(hf_key.startswith(p) for p in skip_patterns):
            continue
        # Try to find this key somewhere in the mapped outputs
        # (keys are not directly comparable — just check none are totally missing)
        mapped_value_match = any(
            torch.equal(sd[hf_key], v)
            for sub in parts.values()
            for v in sub.values()
            if v.shape == sd[hf_key].shape
        )
        if not mapped_value_match:
            missing.append(hf_key)

    if missing and verbose:
        print(f"WARNING: {len(missing)} HF keys not found in any neuron module:")
        for k in missing[:20]:
            print(f"  {k}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    return len(missing) == 0


def save_neuron_checkpoints(
    parts: Dict[str, Dict[str, torch.Tensor]],
    output_dir: str,
) -> None:
    """
    Save each sub-dict to <output_dir>/<name>.pt.

    Parameters
    ----------
    parts:
        Output of convert_hf_to_neuron_state_dicts.
    output_dir:
        Directory to write .pt files into.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    for name, sub_sd in parts.items():
        path = os.path.join(output_dir, f"{name}.pt")
        torch.save(sub_sd, path)
        total = sum(t.numel() for t in sub_sd.values())
        print(f"  Saved {path}  ({len(sub_sd)} tensors, {total/1e6:.1f}M params)")
