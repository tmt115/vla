"""
Unit test for NeuronSmolVLMDecoderLayer.

Compares NxDI block output against transformers LlamaDecoderLayer reference
for SmolVLM2-500M text model architecture.
"""

import sys
sys.path.insert(0, '/home/ubuntu/smol_port/tests')
import _transformers_compat  # bypass huggingface-hub version check — must be first

import math
import torch
import torch.nn as nn
from block_testing_utils import test_block_correctness

# NxDI block under test
sys.path.insert(0, '/home/ubuntu/smol_port/b_vlm_decoder_layer')
from nxdi_block import NeuronSmolVLMDecoderLayer

# Transformers reference
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig

# -----------------------------------------------------------------------
# Architecture constants (SmolVLM2-500M text model)
# -----------------------------------------------------------------------
HIDDEN_SIZE = 960
INTERMEDIATE_SIZE = 2560
NUM_HEADS = 15
NUM_KV_HEADS = 5
HEAD_DIM = 64
MAX_POS = 8192
RMS_NORM_EPS = 1e-5
ROPE_THETA = 100000.0
SEQ_LEN = 241
BATCH_SIZE = 1

# -----------------------------------------------------------------------
# Reference LlamaDecoderLayer wrapper
# -----------------------------------------------------------------------

class WrappedLlamaDecoderLayer(nn.Module):
    """
    Wraps LlamaDecoderLayer so it accepts a single hidden_states tensor
    and returns a single hidden_states tensor.

    Internally creates position_embeddings and a causal attention mask
    matching our prefix-pass inputs.
    """

    def __init__(self):
        super().__init__()
        llama_cfg = LlamaConfig(
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            num_attention_heads=NUM_HEADS,
            num_key_value_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            max_position_embeddings=MAX_POS,
            rms_norm_eps=RMS_NORM_EPS,
            rope_theta=ROPE_THETA,
            hidden_act='silu',
            attention_bias=False,
        )
        llama_cfg._attn_implementation = "eager"
        self.decoder_layer = LlamaDecoderLayer(llama_cfg, layer_idx=0)
        self.rotary_emb = LlamaRotaryEmbedding(config=llama_cfg)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, S, H] bf16
        Returns: [B, S, H] bf16
        """
        B, S, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Build position_ids [B, S]
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        # Build causal attention mask [B, 1, S, S] (True=attend, 0.0/-inf style)
        # LlamaAttention uses additive mask where 0=attend, -inf=mask
        # We pass None and let it use causal masking internally (or build an explicit one)
        # Actually, transformers eager_attention_forward expects the 4D additive float mask
        # where attended positions are 0 and masked are -inf
        causal_mask = torch.zeros(B, 1, S, S, dtype=dtype, device=device)
        # Upper triangle (excluding diagonal) = masked
        mask_upper = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)
        causal_mask[:, :, mask_upper] = float('-inf')

        # Compute position embeddings (cos, sin)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Run the decoder layer
        output = self.decoder_layer(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        # LlamaDecoderLayer returns a tuple or tensor; extract hidden states
        if isinstance(output, (tuple, list)):
            return output[0]
        return output


# -----------------------------------------------------------------------
# Weight mapping: LlamaDecoderLayer -> NeuronSmolVLMDecoderLayer
# -----------------------------------------------------------------------

WEIGHT_MAPPING = {
    # Attention projections (via decoder_layer prefix from wrapper)
    "decoder_layer.self_attn.q_proj.weight": "self_attn.q_proj.weight",
    "decoder_layer.self_attn.k_proj.weight": "self_attn.k_proj.weight",
    "decoder_layer.self_attn.v_proj.weight": "self_attn.v_proj.weight",
    "decoder_layer.self_attn.o_proj.weight": "self_attn.o_proj.weight",
    # MLP projections
    "decoder_layer.mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "decoder_layer.mlp.up_proj.weight": "mlp.up_proj.weight",
    "decoder_layer.mlp.down_proj.weight": "mlp.down_proj.weight",
    # Layer norms
    "decoder_layer.input_layernorm.weight": "input_layernorm.weight",
    "decoder_layer.post_attention_layernorm.weight": "post_attention_layernorm.weight",
}


# -----------------------------------------------------------------------
# Test inputs
# -----------------------------------------------------------------------

def make_inputs(seed=42):
    torch.manual_seed(seed)
    hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)

    # Causal bool mask [B, S, S]: True=attend
    causal_bool = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
    attention_mask = causal_bool.unsqueeze(0).expand(BATCH_SIZE, -1, -1)  # [B, S, S]

    position_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)  # [B, S]

    return hidden_states, attention_mask, position_ids


# -----------------------------------------------------------------------
# Main test
# -----------------------------------------------------------------------

def main():
    hidden_states, attention_mask, position_ids = make_inputs(seed=42)

    # Example inputs (zero-initialized for XLA compilation)
    example_hs = torch.zeros_like(hidden_states)
    example_mask = torch.ones(BATCH_SIZE, SEQ_LEN, SEQ_LEN, dtype=torch.bool)
    example_pos = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.int64)

    test_block_correctness(
        neuron_block_class=NeuronSmolVLMDecoderLayer,
        pytorch_block_class=WrappedLlamaDecoderLayer,
        weight_mapping=WEIGHT_MAPPING,
        neuron_init_kwargs={
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "max_position_embeddings": MAX_POS,
            "rms_norm_eps": RMS_NORM_EPS,
            "rope_theta": ROPE_THETA,
            "attention_bias": False,
        },
        pytorch_init_kwargs={},
        example_inputs=[(example_hs, example_mask, example_pos)],
        test_inputs=[(hidden_states, attention_mask, position_ids)],
        # reference takes single arg hidden_states; the wrapper produces reference output
        reference_inputs=[(hidden_states,)],
        checkpoint_name="b_vlm_decoder_layer.pt",
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        verbose=True,
    )
    print("Test passed!")


if __name__ == "__main__":
    main()
