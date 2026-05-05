"""
prefix_encoder.py — π0 Gemma 2B prefix encoder for Trainium.

Runs the PaliGemma Gemma 2B language model over the 816-token prefix
(3 cameras × 256 tokens + 48 language tokens) with full bidirectional attention
and collects per-layer KV tensors for the suffix denoiser.

Input:  prefix_embs [B, seq_len, 2048] bfloat16
Output: prefix_kv   [num_layers, 2, B, seq_len, 1, 256] bfloat16
          dim layout: [layers, {k=0, v=1}, batch, seq, kv_heads, head_dim]
"""

import os
import torch
import torch.nn as nn
from transformers.models.gemma.modeling_gemma import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from config_constants import (
    VLM_HIDDEN_SIZE, VLM_NUM_LAYERS, VLM_NUM_HEADS,
    VLM_NUM_KV_HEADS, VLM_HEAD_DIM, VLM_INTERMEDIATE_SIZE,
    VLM_RMS_NORM_EPS, PREFIX_LEN,
)


class NeuronPrefixEncoder(nn.Module):
    """
    Gemma 2B prefix encoder with manual layer iteration.
    Collects K and V from each attention layer into a static output tensor.

    Does NOT call PiGemmaModel.forward() — that function uses DynamicCache and
    create_causal_mask (both dynamic, incompatible with Neuron static graphs).
    Instead we iterate layers manually, computing K/V from projections directly.
    """

    def __init__(self, gemma_model, seq_len: int = PREFIX_LEN):
        super().__init__()
        self.layers = gemma_model.layers          # ModuleList of PiGemmaDecoderLayer
        self.rotary_emb = gemma_model.rotary_emb
        self.seq_len = seq_len
        self.num_layers = len(self.layers)
        head_dim = self.layers[0].self_attn.head_dim

        # ── Static buffers ──────────────────────────────────────────────────
        # position_ids: [1, seq_len], values 0..seq_len-1
        self.register_buffer("prefix_position_ids", torch.arange(seq_len).unsqueeze(0))

        # Bidirectional attention mask [1, 1, seq_len, seq_len]:
        # All prefix att_masks = 0 → all tokens attend to all → additive mask = 0.0
        self.register_buffer(
            "prefix_attn_mask",
            torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32),
        )

    def forward(self, prefix_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prefix_embs: [B, seq_len, 2048] bfloat16

        Returns:
            prefix_kv: [num_layers, 2, B, seq_len, kv_heads=1, head_dim=256] bfloat16
        """
        B, seq, hidden = prefix_embs.shape
        hidden_states = prefix_embs

        # Compute rotary embeddings: returns (cos, sin) each [B, seq, head_dim]
        cos, sin = self.rotary_emb(hidden_states, self.prefix_position_ids)
        position_embeddings = (cos, sin)

        kv_list = []
        for layer in self.layers:
            attn = layer.self_attn
            head_dim = attn.head_dim                            # 256
            num_q  = attn.q_proj.weight.shape[0] // head_dim  # 8
            num_kv = attn.k_proj.weight.shape[0] // head_dim  # 1 for MQA

            # ── input layernorm (PiGemmaRMSNorm returns (normed, gate_or_None)) ──
            normed, _ = layer.input_layernorm(hidden_states, cond=None)
            w = normed.dtype

            # ── Q, K, V projections ──────────────────────────────────────────
            # Cast normed to the projection weight dtype (float32 for fresh model, bf16 after loading)
            proj_w = attn.q_proj.weight.dtype
            normed_proj = normed.to(proj_w)
            q = attn.q_proj(normed_proj).view(B, seq, num_q,  head_dim).transpose(1, 2)
            k = attn.k_proj(normed_proj).view(B, seq, num_kv, head_dim).transpose(1, 2)
            v = attn.v_proj(normed_proj).view(B, seq, num_kv, head_dim).transpose(1, 2)

            # ── Apply RoPE ───────────────────────────────────────────────────
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # ── Collect K, V before attention ────────────────────────────────
            # Store as [B, seq, kv_heads, head_dim] = [B, seq, 1, 256]
            k_stored = k.permute(0, 2, 1, 3)   # [B, 1, seq, 256] → [B, seq, 1, 256]
            v_stored = v.permute(0, 2, 1, 3)
            kv_list.append(torch.stack([k_stored, v_stored], dim=0))  # [2, B, seq, 1, 256]

            # ── Attention (full bidirectional over prefix) ────────────────────
            # eager_attention_forward(module, q, k, v, mask, scaling, ...)
            attn_mask_bf16 = self.prefix_attn_mask.to(w)
            attn_out, _ = eager_attention_forward(
                attn, q, k, v, attn_mask_bf16, scaling=attn.scaling,
            )
            attn_out = attn_out.reshape(B, seq, num_q * head_dim)

            # ── Output projection ────────────────────────────────────────────
            if attn_out.dtype != attn.o_proj.weight.dtype:
                attn_out = attn_out.to(attn.o_proj.weight.dtype)
            attn_out = attn.o_proj(attn_out)

            # ── First residual ───────────────────────────────────────────────
            hidden_states = hidden_states + attn_out

            # ── Post-attention layernorm ─────────────────────────────────────
            normed2, _ = layer.post_attention_layernorm(hidden_states, cond=None)
            if normed2.dtype != layer.mlp.up_proj.weight.dtype:
                normed2 = normed2.to(layer.mlp.up_proj.weight.dtype)

            # ── MLP ──────────────────────────────────────────────────────────
            mlp_out = layer.mlp(normed2)

            # ── Second residual ──────────────────────────────────────────────
            hidden_states = hidden_states + mlp_out

        # Stack: [num_layers, 2, B, seq, 1, 256]
        return torch.stack(kv_list, dim=0)


def build_gemma_model(hidden_size=VLM_HIDDEN_SIZE, num_layers=VLM_NUM_LAYERS,
                      num_heads=VLM_NUM_HEADS, kv_heads=VLM_NUM_KV_HEADS,
                      head_dim=VLM_HEAD_DIM, intermediate_size=VLM_INTERMEDIATE_SIZE,
                      rms_norm_eps=VLM_RMS_NORM_EPS):
    """Instantiate a PiGemmaModel with given dimensions."""
    from transformers.models.gemma.modeling_gemma import GemmaConfig
    from lerobot.policies.pi_gemma import PiGemmaModel

    cfg = GemmaConfig(
        head_dim=head_dim,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_hidden_layers=num_layers,
        num_key_value_heads=kv_heads,
        vocab_size=257152,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=rms_norm_eps,
        use_adarms=False,
        adarms_cond_dim=None,
    )
    cfg._attn_implementation = "eager"
    return PiGemmaModel(cfg)


def load_prefix_encoder(checkpoint_path: str, seq_len: int = PREFIX_LEN) -> NeuronPrefixEncoder:
    """Load Gemma 2B backbone weights from the lerobot/pi0 checkpoint."""
    from safetensors.torch import load_file

    gemma = build_gemma_model()

    sd = load_file(checkpoint_path)

    # Keys in checkpoint: model.paligemma_with_expert.paligemma.model.language_model.layers.*
    # Note: NO extra .model. between language_model and layers
    src_prefix = "model.paligemma_with_expert.paligemma.model.language_model."
    target_sd = {}
    for k, v in sd.items():
        if k.startswith(src_prefix):
            new_k = k[len(src_prefix):]
            # Skip embed_tokens — prefix encoder receives pre-embedded inputs
            if new_k.startswith("embed_tokens"):
                continue
            target_sd[new_k] = v

    missing, unexpected = gemma.load_state_dict(target_sd, strict=False)
    non_embed = [k for k in missing if "embed_tokens" not in k]
    if non_embed:
        print(f"load_prefix_encoder: {len(non_embed)} unexpected missing keys: {non_embed[:5]}")

    encoder = NeuronPrefixEncoder(gemma, seq_len=seq_len)
    return encoder


def compile_prefix_encoder(encoder: NeuronPrefixEncoder, save_path: str) -> None:
    import torch_neuronx
    os.makedirs(save_path, exist_ok=True)
    neff_path = os.path.join(save_path, "model.pt")
    if os.path.exists(neff_path):
        print(f"Already compiled, skipping: {neff_path}")
        return
    example = torch.zeros(1, encoder.seq_len, VLM_HIDDEN_SIZE, dtype=torch.bfloat16)
    encoder.eval()
    traced = torch_neuronx.trace(
        encoder,
        example,
        compiler_workdir=os.path.join(save_path, "workdir/"),
        compiler_args=["-O1", "--auto-cast=none"],
    )
    torch.jit.save(traced, neff_path)
    sz = os.path.getsize(neff_path) / 1e6
    print(f"Saved: {neff_path} ({sz:.1f} MB)")
