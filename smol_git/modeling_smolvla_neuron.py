"""
NeuronSmolVLAPrefix  and  NeuronSmolVLADenoiseStep
===================================================

Two torch_neuronx-compiled modules for SmolVLA inference on Trainium.

Inference pipeline
------------------
  1.  [once]  NeuronSmolVLAPrefix.forward(prefix_embs, attn_mask, pos_ids)
              → tuple of 32 KV tensors  ([1,241,5,64] bf16 each)

  2.  [×10 Euler steps]
              NeuronSmolVLADenoiseStep.forward(noisy_actions, timestep, *kv_tensors)
              → v_t  [1, 50, 32]  float32

Architecture notes
------------------
  VLM:    hidden=960,  heads=15, kv_heads=5, head_dim=64, layers=16
  Expert: hidden=720,  heads=15, kv_heads=5, head_dim=64, layers=16
  prefix_len=241  suffix_len=50  full_len=291

Expert layer alternation (from SmolVLMWithExpertModel, self_attn_every_n_layers=2):
  Even layers (0,2,...,14):  "self-attn mode"
    Q/K/V from suffix_embs.
    K = concat(prefix_vlm_K, suffix_K)  — 291 tokens total.
    V = concat(prefix_vlm_V, suffix_V).
    RoPE on Q and K with suffix positions (241-290).
    Attention: Q[50] × K[291], mask [B,50,291].

  Odd layers  (1,3,...,15):  "cross-attn mode"
    Q from expert q_proj.
    K = expert k_proj(prefix_vlm_K.reshape(B,241,320))  — 241 tokens.
    V = expert v_proj(prefix_vlm_V.reshape(B,241,320)).
    RoPE on Q only with zero-based positions (0-49).
    No RoPE on K/V (VLM KV was already RoPE'd in prefix pass).
    Attention: Q[50] × K[241], mask [B,50,241].
"""

import importlib.util
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Import Phase-2 building blocks
# Each nxdi_block.py lives in its own directory but shares the same module
# name; use importlib to load each under a unique alias so Python's module
# cache doesn't return the wrong file.
# --------------------------------------------------------------------------- #

def _load_nxdi_block(directory: str, alias: str):
    """Load <directory>/nxdi_block.py and register it as sys.modules[alias]."""
    spec = importlib.util.spec_from_file_location(
        alias, directory + '/nxdi_block.py'
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_b = _load_nxdi_block('/home/ubuntu/smol_port/b_vlm_decoder_layer', 'nxdi_block_b')
NeuronSmolVLMDecoderLayer = _b.NeuronSmolVLMDecoderLayer

_c = _load_nxdi_block('/home/ubuntu/smol_port/c_expert_layer', 'nxdi_block_c')
NeuronSmolVLAExpertLayer  = _c.NeuronSmolVLAExpertLayer
_RoPE                     = _c._RoPE
_apply_rotary             = _c._apply_rotary

_e = _load_nxdi_block('/home/ubuntu/smol_port/e_action_projections', 'nxdi_block_e')
NeuronSmolVLAActionProjections = _e.NeuronSmolVLAActionProjections


# --------------------------------------------------------------------------- #
# Shape constants (SmolVLA-base defaults)
# --------------------------------------------------------------------------- #
_VLM_HIDDEN     = 960
_VLM_INTERMED   = 2560
_VLM_NUM_HEADS  = 15
_VLM_KV_HEADS   = 5
_VLM_HEAD_DIM   = 64
_VLM_MAX_POS    = 8192
_VLM_RMS_EPS    = 1e-5
_VLM_ROPE_THETA = 100000.0
_VLM_LAYERS     = 16

_EXP_HIDDEN    = 720
_EXP_NUM_HEADS = 15
_EXP_KV_HEADS  = 5
_EXP_HEAD_DIM  = 64
_VLM_KV_DIM    = _EXP_KV_HEADS * _EXP_HEAD_DIM  # 320

_PREFIX_LEN = 241
_SUFFIX_LEN = 50
_FULL_LEN   = _PREFIX_LEN + _SUFFIX_LEN          # 291

_MAX_STATE_DIM  = 32
_MAX_ACTION_DIM = 32


# =========================================================================== #
# NeuronSmolVLAPrefix
# =========================================================================== #

class NeuronSmolVLAPrefix(nn.Module):
    """
    VLM prefix pass: 16 Llama decoder layers that produce the KV cache.

    Input
    -----
    prefix_embs : [1, 241, 960]  bf16
        Combined VLM token embeddings (image patches + text tokens).
    attn_mask   : [1, 241, 241]  bool  (True = attend)
        Causal / padding mask for the prefix.
    pos_ids     : [1, 241]       int64

    Output
    ------
    Flat tuple of 32 tensors (16 layers × 2):
        kv_0_key, kv_0_val, kv_1_key, kv_1_val, ..., kv_15_key, kv_15_val
    Each tensor:  [1, 241, 5, 64]  bf16
    """

    def __init__(self, config=None):
        super().__init__()
        self.layers = nn.ModuleList([
            NeuronSmolVLMDecoderLayer(
                hidden_size          = _VLM_HIDDEN,
                intermediate_size    = _VLM_INTERMED,
                num_attention_heads  = _VLM_NUM_HEADS,
                num_key_value_heads  = _VLM_KV_HEADS,
                head_dim             = _VLM_HEAD_DIM,
                max_position_embeddings = _VLM_MAX_POS,
                rms_norm_eps         = _VLM_RMS_EPS,
                rope_theta           = _VLM_ROPE_THETA,
                attention_bias       = False,
            )
            for _ in range(_VLM_LAYERS)
        ])

    def forward(
        self,
        prefix_embs: torch.Tensor,   # [1, 241, 960]  bf16
        attn_mask:   torch.Tensor,   # [1, 241, 241]  bool
        pos_ids:     torch.Tensor,   # [1, 241]       int64
    ):
        hidden = prefix_embs
        kv_out = []
        for layer in self.layers:
            hidden, k, v = layer(hidden, attn_mask, pos_ids)
            kv_out.append(k)   # [1, 241, 5, 64] bf16
            kv_out.append(v)   # [1, 241, 5, 64] bf16
        # Return as flat tuple — required by torch_neuronx.trace
        return tuple(kv_out)


# =========================================================================== #
# NeuronSmolVLADenoiseStep
# =========================================================================== #

class NeuronSmolVLADenoiseStep(nn.Module):
    """
    Single Euler denoising step.  Called 10× per inference.

    Input
    -----
    noisy_actions : [1, 50, 32]  float32
    timestep      : [1]          float32
    kv_0_key … kv_15_val : 32 tensors,  each [1, 241, 5, 64]  bf16
        (alternating key/value per VLM layer, same order as NeuronSmolVLAPrefix output)

    Output
    ------
    v_t : [1, 50, 32]  float32   (velocity prediction)
    """

    def __init__(self, config=None):
        super().__init__()

        # Action projection layers (state_proj, action_in_proj, action_out_proj,
        # action_time_mlp_in, action_time_mlp_out  +  sine_scale buffer)
        self.projections = NeuronSmolVLAActionProjections(config=config)

        # 16 expert layers: even indices = self-attn mode, odd = cross-attn mode
        self.expert_layers = nn.ModuleList([
            NeuronSmolVLAExpertLayer(
                config            = config,
                is_self_attn_layer = (i % 2 == 0),
            )
            for i in range(_VLM_LAYERS)
        ])

        # Precompute static tensors to avoid torch.arange inside XLA trace.
        # Shapes are batch-1; will be expanded in forward.
        suffix_pos = torch.arange(_SUFFIX_LEN, dtype=torch.int64) + _PREFIX_LEN  # [50] vals 241-290
        zero_pos   = torch.arange(_SUFFIX_LEN, dtype=torch.int64)                # [50] vals 0-49
        # Full attention mask  [1, 50, 291]  — suffix attends to prefix+suffix
        full_mask  = torch.ones(1, _SUFFIX_LEN, _FULL_LEN,   dtype=torch.bool)
        # Cross-attn mask      [1, 50, 241]  — suffix attends to prefix only
        cross_mask = torch.ones(1, _SUFFIX_LEN, _PREFIX_LEN, dtype=torch.bool)

        self.register_buffer("suffix_pos", suffix_pos, persistent=False)
        self.register_buffer("zero_pos",   zero_pos,   persistent=False)
        self.register_buffer("full_mask",  full_mask,  persistent=False)
        self.register_buffer("cross_mask", cross_mask, persistent=False)

    # ---------------------------------------------------------------------- #
    # Per-layer helpers
    # ---------------------------------------------------------------------- #

    def _self_attn_denoise(
        self,
        layer:        NeuronSmolVLAExpertLayer,
        hidden:       torch.Tensor,   # [B, 50, 720]  bf16
        prefix_k:     torch.Tensor,   # [B, 241, 5, 64]  bf16  (VLM RoPE'd at pos 0-240)
        prefix_v:     torch.Tensor,   # [B, 241, 5, 64]  bf16
        suffix_pos_ids: torch.Tensor, # [B, 50]  int64  vals 241-290
        full_mask:    torch.Tensor,   # [B, 50, 291]  bool
    ) -> torch.Tensor:
        """
        Self-attn expert layer (even indices).

        Q/K/V from suffix_embs.
        K_full = concat(prefix_vlm_K, suffix_K)  →  291 tokens.
        V_full = concat(prefix_vlm_V, suffix_V).
        RoPE on Q and K with positions 241-290.
        """
        B, S_q, _ = hidden.shape
        attn = layer.self_attn
        nH, nKV, hD = attn.NUM_HEADS, attn.NUM_KV_HEADS, attn.HEAD_DIM

        # --- LayerNorm + QKV projection ---
        normed = layer.input_layernorm(hidden)                             # [B,50,720]
        q = attn.q_proj(normed).view(B, S_q, nH, hD).transpose(1, 2)     # [B,15,50,64]
        k = attn.k_proj(normed).view(B, S_q, nKV, hD).transpose(1, 2)    # [B, 5,50,64]
        v = attn.v_proj(normed).view(B, S_q, nKV, hD).transpose(1, 2)    # [B, 5,50,64]

        # --- RoPE on Q and K with suffix positions 241-290 ---
        cos, sin = attn.rotary_emb(q, suffix_pos_ids)           # cos/sin: [B,50,64]
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        # --- Concatenate prefix VLM KV (already RoPE'd at pos 0-240) ---
        # prefix_k: [B, 241, 5, 64]  → transpose to [B, 5, 241, 64]
        pk = prefix_k.to(dtype=hidden.dtype).transpose(1, 2)              # [B, 5,241,64]
        pv = prefix_v.to(dtype=hidden.dtype).transpose(1, 2)              # [B, 5,241,64]
        k_full = torch.cat([pk, k], dim=2)                                 # [B, 5,291,64]
        v_full = torch.cat([pv, v], dim=2)                                 # [B, 5,291,64]

        # --- GQA expand: 5 KV-heads → 15 Q-heads (n_rep=3) ---
        n_rep = nH // nKV
        k_full = (k_full.unsqueeze(2)
                        .expand(B, nKV, n_rep, _FULL_LEN, hD)
                        .reshape(B, nH, _FULL_LEN, hD))
        v_full = (v_full.unsqueeze(2)
                        .expand(B, nKV, n_rep, _FULL_LEN, hD)
                        .reshape(B, nH, _FULL_LEN, hD))

        # --- Scaled dot-product attention ---
        scale  = hD ** -0.5
        scores = torch.matmul(q, k_full.transpose(2, 3)) * scale          # [B,15,50,291]
        mask   = full_mask.unsqueeze(1)                                    # [B, 1,50,291]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs  = F.softmax(scores.float(), dim=-1).to(hidden.dtype)
        attn_out = torch.matmul(probs, v_full)                             # [B,15,50,64]

        # --- Output projection + first residual ---
        attn_out = attn_out.transpose(1, 2).reshape(B, S_q, nH * hD)     # [B,50,960]
        attn_out = attn.o_proj(attn_out)                                   # [B,50,720]
        hidden   = hidden + attn_out

        # --- MLP sub-layer ---
        residual = hidden
        hidden   = layer.post_attention_layernorm(hidden)
        hidden   = layer.mlp(hidden)
        hidden   = residual + hidden

        return hidden                                                        # [B,50,720]

    def _cross_attn_denoise(
        self,
        layer:        NeuronSmolVLAExpertLayer,
        hidden:       torch.Tensor,   # [B, 50, 720]  bf16
        prefix_k:     torch.Tensor,   # [B, 241, 5, 64]  bf16
        prefix_v:     torch.Tensor,   # [B, 241, 5, 64]  bf16
        zero_pos_ids: torch.Tensor,   # [B, 50]  int64  vals 0-49
        cross_mask:   torch.Tensor,   # [B, 50, 241]  bool
    ) -> torch.Tensor:
        """
        Cross-attn expert layer (odd indices).

        Q from expert q_proj (RoPE with zero-based positions).
        K/V from VLM prefix via expert cross-attn k_proj/v_proj (320→320).
        No RoPE on K/V.
        """
        B, S_q, _ = hidden.shape
        attn = layer.self_attn    # NeuronSmolVLAExpertAttention(is_cross_attn=True)
        S_kv = prefix_k.shape[1]  # 241
        nH, nKV, hD = attn.NUM_HEADS, attn.NUM_KV_HEADS, attn.HEAD_DIM

        # --- LayerNorm + Q projection ---
        normed = layer.input_layernorm(hidden)                             # [B,50,720]
        q = attn.q_proj(normed).view(B, S_q, nH, hD).transpose(1, 2)     # [B,15,50,64]

        # --- K/V from VLM prefix via expert cross-attn projections ---
        # prefix_k: [B, 241, 5, 64]  → flatten to [B, 241, 320]
        kv_k_flat = prefix_k.to(dtype=hidden.dtype).reshape(B, S_kv, _VLM_KV_DIM)
        kv_v_flat = prefix_v.to(dtype=hidden.dtype).reshape(B, S_kv, _VLM_KV_DIM)
        k = attn.k_proj(kv_k_flat).view(B, S_kv, nKV, hD).transpose(1, 2)   # [B, 5,241,64]
        v = attn.v_proj(kv_v_flat).view(B, S_kv, nKV, hD).transpose(1, 2)   # [B, 5,241,64]

        # --- RoPE on Q only, zero-based positions ---
        cos, sin = attn.rotary_emb(q, zero_pos_ids)             # cos/sin: [B,50,64]
        q = _apply_rotary(q, cos, sin)

        # --- GQA expand: 5 → 15 ---
        n_rep = nH // nKV
        k = (k.unsqueeze(2)
              .expand(B, nKV, n_rep, S_kv, hD)
              .reshape(B, nH, S_kv, hD))
        v = (v.unsqueeze(2)
              .expand(B, nKV, n_rep, S_kv, hD)
              .reshape(B, nH, S_kv, hD))

        # --- Scaled dot-product attention ---
        scale  = hD ** -0.5
        scores = torch.matmul(q, k.transpose(2, 3)) * scale               # [B,15,50,241]
        mask   = cross_mask.unsqueeze(1)                                   # [B, 1,50,241]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs  = F.softmax(scores.float(), dim=-1).to(hidden.dtype)
        attn_out = torch.matmul(probs, v)                                  # [B,15,50,64]

        # --- Output projection + first residual ---
        attn_out = attn_out.transpose(1, 2).reshape(B, S_q, nH * hD)     # [B,50,960]
        attn_out = attn.o_proj(attn_out)                                   # [B,50,720]
        hidden   = hidden + attn_out

        # --- MLP sub-layer ---
        residual = hidden
        hidden   = layer.post_attention_layernorm(hidden)
        hidden   = layer.mlp(hidden)
        hidden   = residual + hidden

        return hidden                                                        # [B,50,720]

    # ---------------------------------------------------------------------- #
    # Main forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        noisy_actions: torch.Tensor,   # [1, 50, 32]  float32
        timestep:      torch.Tensor,   # [1]          float32
        *kv_tensors,                   # 32 tensors:  each [1,241,5,64] bf16
    ) -> torch.Tensor:                 # v_t  [1, 50, 32]  float32
        B      = noisy_actions.shape[0]
        device = noisy_actions.device

        # ---- 1. Compute suffix embeddings --------------------------------- #
        # Cast to bfloat16 so expert layers stay in bf16 arithmetic.
        suffix_embs = self.projections.forward_suffix(
            noisy_actions.to(dtype=torch.bfloat16), timestep
        )   # [B, 50, 720]  bf16

        # ---- 2. Unpack KV tensor pairs ------------------------------------ #
        # kv_tensors layout: k0, v0, k1, v1, ..., k15, v15
        prefix_kvs = [
            (kv_tensors[2 * i], kv_tensors[2 * i + 1])
            for i in range(_VLM_LAYERS)
        ]

        # ---- 3. Expand precomputed static buffers to batch size ----------- #
        suffix_pos_ids = self.suffix_pos.unsqueeze(0).expand(B, -1)   # [B, 50]
        zero_pos_ids   = self.zero_pos.unsqueeze(0).expand(B, -1)     # [B, 50]
        full_mask      = self.full_mask.expand(B, -1, -1)             # [B, 50, 291]
        cross_mask     = self.cross_mask.expand(B, -1, -1)            # [B, 50, 241]

        # ---- 4. Run expert layers ----------------------------------------- #
        hidden = suffix_embs
        for i, layer in enumerate(self.expert_layers):
            prefix_k, prefix_v = prefix_kvs[i]
            if i % 2 == 0:
                hidden = self._self_attn_denoise(
                    layer, hidden, prefix_k, prefix_v,
                    suffix_pos_ids, full_mask,
                )
            else:
                hidden = self._cross_attn_denoise(
                    layer, hidden, prefix_k, prefix_v,
                    zero_pos_ids, cross_mask,
                )

        # ---- 5. Project expert output → action space ---------------------- #
        v_t = self.projections.action_out_proj(hidden)   # [B, 50, 32]  bf16
        return v_t.to(dtype=torch.float32)               # [B, 50, 32]  float32
