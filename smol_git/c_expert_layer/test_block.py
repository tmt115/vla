"""
Unit test for NeuronSmolVLAExpertLayer.

Tests both modes:
  - Self-attention (is_self_attn_layer=True, even layers)
  - Cross-attention (is_self_attn_layer=False, odd layers)

Reference: LlamaDecoderLayer from transformers, with k_proj/v_proj patched for
cross-attn layers to match SmolVLMWithExpertModel.__init__ lines 111-122.
"""
import sys
sys.path.insert(0, '/home/ubuntu/smol_port/tests')
import _transformers_compat  # bypass huggingface-hub version check — must be first

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/ubuntu/smol_port/c_expert_layer')
from nxdi_block import NeuronSmolVLAExpertLayer

from block_testing_utils import test_block_correctness

# Import reference directly from original source — not from any local file
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.configuration_llama import LlamaConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERT_HIDDEN   = 720
EXPERT_INTERMED = 2048
NUM_HEADS       = 15
NUM_KV_HEADS    = 5
HEAD_DIM        = 64
SUFFIX_LEN      = 50   # chunk_size
PREFIX_LEN      = 241  # VLM prefix tokens
FULL_LEN        = PREFIX_LEN + SUFFIX_LEN  # 291
BATCH_SIZE      = 1
RMS_NORM_EPS    = 1e-5
ROPE_THETA      = 100000.0
VLM_KV_DIM     = NUM_KV_HEADS * HEAD_DIM  # 320

# ---------------------------------------------------------------------------
# Build base expert LlamaConfig
# ---------------------------------------------------------------------------
def _expert_llama_cfg():
    cfg = LlamaConfig(
        hidden_size=EXPERT_HIDDEN,
        intermediate_size=EXPERT_INTERMED,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=8192,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=ROPE_THETA,
        hidden_act='silu',
        attention_bias=False,
    )
    cfg._attn_implementation = "eager"
    return cfg


# ===========================================================================
# TEST 1: Self-attention mode (even layers)
# ===========================================================================

class SelfAttnRefWrapper(nn.Module):
    """
    Wraps LlamaDecoderLayer to accept a single hidden_states tensor and return
    a single hidden_states tensor (self-attention only, causal mask).
    """
    def __init__(self):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        cfg = _expert_llama_cfg()
        self.layer = LlamaDecoderLayer(cfg, layer_idx=0)
        self.rotary_emb = LlamaRotaryEmbedding(config=cfg)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        # Full (non-causal) attention mask — expert during denoise sees all suffix tokens
        attn_mask = torch.zeros(B, 1, S, S, dtype=dtype, device=device)
        pos_emb = self.rotary_emb(hidden_states, position_ids)
        out = self.layer(
            hidden_states=hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            position_embeddings=pos_emb,
        )
        return out[0] if isinstance(out, (tuple, list)) else out


SELF_ATTN_WEIGHT_MAP = {
    "layer.self_attn.q_proj.weight": "self_attn.q_proj.weight",
    "layer.self_attn.k_proj.weight": "self_attn.k_proj.weight",
    "layer.self_attn.v_proj.weight": "self_attn.v_proj.weight",
    "layer.self_attn.o_proj.weight": "self_attn.o_proj.weight",
    "layer.mlp.gate_proj.weight":    "mlp.gate_proj.weight",
    "layer.mlp.up_proj.weight":      "mlp.up_proj.weight",
    "layer.mlp.down_proj.weight":    "mlp.down_proj.weight",
    "layer.input_layernorm.weight":          "input_layernorm.weight",
    "layer.post_attention_layernorm.weight": "post_attention_layernorm.weight",
}


def test_self_attn_mode():
    B, S = BATCH_SIZE, SUFFIX_LEN
    torch.manual_seed(0)
    hidden_states   = torch.randn(B, S, EXPERT_HIDDEN, dtype=torch.bfloat16)
    # Full bool mask (all tokens attend to all)
    attn_mask_bool  = torch.ones(B, S, S, dtype=torch.bool)
    position_ids    = torch.arange(S).unsqueeze(0).expand(B, -1)

    example_hs   = torch.zeros(B, S, EXPERT_HIDDEN, dtype=torch.bfloat16)
    example_mask = torch.ones(B, S, S, dtype=torch.bool)
    example_pos  = torch.zeros(B, S, dtype=torch.int64)

    test_block_correctness(
        neuron_block_class=NeuronSmolVLAExpertLayer,
        pytorch_block_class=SelfAttnRefWrapper,
        weight_mapping=SELF_ATTN_WEIGHT_MAP,
        neuron_init_kwargs={"is_self_attn_layer": True},
        pytorch_init_kwargs={},
        example_inputs=[(example_hs, example_mask, example_pos)],
        test_inputs=[(hidden_states, attn_mask_bool, position_ids)],
        reference_inputs=[(hidden_states,)],
        checkpoint_name="c_expert_self_attn.pt",
        batch_size=B,
        seq_len=S,
        hidden_size=EXPERT_HIDDEN,
        verbose=True,
    )
    print("PASSED: Expert self-attn mode")


# ===========================================================================
# TEST 2: Cross-attention mode (odd layers)
# ===========================================================================

class CrossAttnRefWrapper(nn.Module):
    """
    Wraps a patched LlamaDecoderLayer to replicate cross-attn mode from
    SmolVLMWithExpertModel.forward_cross_attn_layer.

    The patched layer has:
      k_proj: Linear(VLM_KV_DIM=320, expert_kv_out=320, bias=False)
      v_proj: Linear(VLM_KV_DIM=320, expert_kv_out=320, bias=False)

    forward(expert_hidden) → expert_hidden   (kv_key/kv_value stored as buffers)

    kv_key and kv_value are passed at construction time and stored as buffers so
    that the harness can call reference_block(expert_hidden) with a single arg.
    """
    def __init__(self, kv_key: torch.Tensor, kv_value: torch.Tensor):
        super().__init__()
        cfg = _expert_llama_cfg()
        self.layer = LlamaDecoderLayer(cfg, layer_idx=0)
        # Patch k/v proj: input is VLM KV flattened dim (320) not expert hidden (720)
        kv_out = NUM_KV_HEADS * HEAD_DIM  # 320
        self.layer.self_attn.k_proj = nn.Linear(VLM_KV_DIM, kv_out, bias=False)
        self.layer.self_attn.v_proj = nn.Linear(VLM_KV_DIM, kv_out, bias=False)
        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HEAD_DIM
        # Store KV tensors as buffers — survive harness .to(bfloat16), not reinitialized
        self.register_buffer("stored_kv_key",   kv_key.clone())
        self.register_buffer("stored_kv_value", kv_value.clone())

    def _rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope_q_only(self, q, position_ids):
        """Apply simple RoPE to Q only (zero-based positions)."""
        B, nh, S, hd = q.shape
        inv_freq = 1.0 / (ROPE_THETA ** (
            torch.arange(0, hd, 2, dtype=torch.float32, device=q.device) / hd
        ))
        pos = position_ids[:, None, :].float()  # [B, 1, S]
        freqs = (inv_freq[None, :, None] * pos).transpose(1, 2)  # [B, S, hd//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, S, hd]
        cos = emb.cos().unsqueeze(1).to(q.dtype)  # [B, 1, S, hd]
        sin = emb.sin().unsqueeze(1).to(q.dtype)
        return (q * cos) + (self._rotate_half(q) * sin)

    def forward(self, expert_hidden: torch.Tensor) -> torch.Tensor:
        """
        expert_hidden: [B, S_q, 720]
        kv_key / kv_value are read from stored buffers (registered in __init__).
        """
        kv_key   = self.stored_kv_key
        kv_value = self.stored_kv_value
        B, S_q, _ = expert_hidden.shape
        S_kv = kv_key.shape[1]
        device = expert_hidden.device
        dtype = expert_hidden.dtype

        layer = self.layer
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        n_rep = nh // nkv  # 3

        # input layernorm
        normed = layer.input_layernorm(expert_hidden)

        # Q from expert hidden
        q = layer.self_attn.q_proj(normed).view(B, S_q, nh, hd).transpose(1, 2)  # [B,15,Sq,64]

        # K/V from VLM KV cache (flattened to [B, Skv, 320])
        kv_key_flat   = kv_key.reshape(B, S_kv, VLM_KV_DIM).to(dtype)
        kv_val_flat   = kv_value.reshape(B, S_kv, VLM_KV_DIM).to(dtype)
        k = layer.self_attn.k_proj(kv_key_flat).view(B, S_kv, nkv, hd).transpose(1, 2)
        v = layer.self_attn.v_proj(kv_val_flat).view(B, S_kv, nkv, hd).transpose(1, 2)

        # RoPE on Q only (zero-based positions)
        position_ids = torch.arange(S_q, device=device).unsqueeze(0).expand(B, -1)
        q = self._apply_rope_q_only(q, position_ids)

        # GQA expand
        k = k.unsqueeze(2).expand(B, nkv, n_rep, S_kv, hd).reshape(B, nh, S_kv, hd)
        v = v.unsqueeze(2).expand(B, nkv, n_rep, S_kv, hd).reshape(B, nh, S_kv, hd)

        # Attention (full, no causal mask for cross-attn)
        scale = hd ** -0.5
        scores = torch.matmul(q, k.transpose(2, 3)) * scale  # [B, 15, Sq, Skv]
        probs = F.softmax(scores.float(), dim=-1).to(dtype)
        attn_out = torch.matmul(probs, v)  # [B, 15, Sq, 64]
        attn_out = attn_out.transpose(1, 2).reshape(B, S_q, nh * hd)
        attn_out = layer.self_attn.o_proj(attn_out)

        # Residual
        hidden = expert_hidden + attn_out

        # MLP
        residual2 = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual2 + hidden

        return hidden


CROSS_ATTN_WEIGHT_MAP = {
    "layer.self_attn.q_proj.weight": "self_attn.q_proj.weight",
    "layer.self_attn.k_proj.weight": "self_attn.k_proj.weight",
    "layer.self_attn.v_proj.weight": "self_attn.v_proj.weight",
    "layer.self_attn.o_proj.weight": "self_attn.o_proj.weight",
    "layer.mlp.gate_proj.weight":    "mlp.gate_proj.weight",
    "layer.mlp.up_proj.weight":      "mlp.up_proj.weight",
    "layer.mlp.down_proj.weight":    "mlp.down_proj.weight",
    "layer.input_layernorm.weight":          "input_layernorm.weight",
    "layer.post_attention_layernorm.weight": "post_attention_layernorm.weight",
}


def _cross_attn_sync_weights(reference_block, checkpoint_path, weight_mapping, verbose):
    """
    Custom sync: CrossAttnRefWrapper has patched k_proj/v_proj (320→320),
    which differ from the standard LlamaDecoderLayer shapes in the checkpoint.
    We need to sync only the shapes that match.
    """
    import torch
    neuron_sd = torch.load(checkpoint_path)
    ref_sd = reference_block.state_dict()

    if verbose:
        print("\nCross-attn weight sync:")
        print("Neuron keys:", sorted(neuron_sd.keys()))
        print("Ref keys:",    sorted(ref_sd.keys()))

    updated = 0
    for ref_key, nxd_key in weight_mapping.items():
        full_nxd_key = f"block.{nxd_key}" if not nxd_key.startswith("block.") else nxd_key
        if ref_key not in ref_sd or full_nxd_key not in neuron_sd:
            if verbose:
                print(f"  SKIP {ref_key} (not found in one side)")
            continue
        rt = ref_sd[ref_key].to(neuron_sd[full_nxd_key].dtype)
        if rt.shape != neuron_sd[full_nxd_key].shape:
            if verbose:
                print(f"  SHAPE MISMATCH {ref_key}: ref={rt.shape} nxd={neuron_sd[full_nxd_key].shape}")
            continue
        neuron_sd[full_nxd_key] = rt.contiguous().clone()
        updated += 1
        if verbose:
            print(f"  ✓ {ref_key} → {full_nxd_key}")

    torch.save(neuron_sd, checkpoint_path)
    return updated


def test_cross_attn_mode():
    B, S_q, S_kv = BATCH_SIZE, SUFFIX_LEN, PREFIX_LEN
    torch.manual_seed(1)
    expert_hidden = torch.randn(B, S_q, EXPERT_HIDDEN, dtype=torch.bfloat16)
    kv_key   = torch.randn(B, S_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    kv_value = torch.randn(B, S_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)

    # Full cross-attn mask: suffix tokens attend to all prefix tokens
    attn_mask = torch.ones(B, S_q, S_kv, dtype=torch.bool)
    position_ids = torch.arange(S_q, dtype=torch.int64).unsqueeze(0).expand(B, -1)

    example_hs   = torch.zeros(B, S_q, EXPERT_HIDDEN, dtype=torch.bfloat16)
    example_kk   = torch.zeros(B, S_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    example_kv   = torch.zeros(B, S_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    example_mask = torch.ones(B, S_q, S_kv, dtype=torch.bool)
    example_pos  = torch.zeros(B, S_q, dtype=torch.int64)

    test_block_correctness(
        neuron_block_class=NeuronSmolVLAExpertLayer,
        pytorch_block_class=CrossAttnRefWrapper,
        weight_mapping=CROSS_ATTN_WEIGHT_MAP,
        neuron_init_kwargs={"is_self_attn_layer": False},
        pytorch_init_kwargs={"kv_key": kv_key, "kv_value": kv_value},
        example_inputs=[(example_hs, example_mask, example_pos, example_kk, example_kv)],
        test_inputs=[(expert_hidden, attn_mask, position_ids, kv_key, kv_value)],
        reference_inputs=[(expert_hidden,)],   # harness calls ref(expert_hidden) only
        checkpoint_name="c_expert_cross_attn.pt",
        batch_size=B,
        seq_len=S_q,
        hidden_size=EXPERT_HIDDEN,
        verbose=True,
        sync_weights_fn=_cross_attn_sync_weights,
    )
    print("PASSED: Expert cross-attn mode")


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Expert self-attention mode")
    print("=" * 60)
    test_self_attn_mode()

    print("\n" + "=" * 60)
    print("TEST 2: Expert cross-attention mode")
    print("=" * 60)
    test_cross_attn_mode()

    print("\nAll expert layer tests PASSED")
