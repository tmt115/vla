"""
Integration smoke test: vision encoder → VLM backbone → action expert.

Tests that the three compilation units connect correctly at their interfaces
and produce plausible outputs end-to-end.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd /home/ubuntu/smol-port-skilled
    python tests/test_integration.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEROBOT_PATH = '/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages'
if LEROBOT_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_PATH)

import torch
import math

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
    STATE_DIM,
    TIMESTEP_EMBED_DIM,
    NUM_CONDITIONING_TOKENS,
    NUM_DENOISING_STEPS,
    MIN_PERIOD,
    MAX_PERIOD,
)
from vision_encoder_block import NeuronVisionEncoderBlock
from vlm_backbone_block import NeuronVLMBackboneBlock
from action_expert_block import NeuronSmolVLADenoisingWrapper


def make_prefix_embeddings(vision_embs, text_embs, state_embs):
    """Concatenate text + vision + state embeddings into prefix [B, 113, 960]."""
    # Matches the order used in SmolVLMWithExpertModel.forward:
    # prefix = [text_tokens | vision_tokens | state_token]
    return torch.cat([text_embs, vision_embs, state_embs], dim=1)


def sinusoidal_timestep_emb(t: float, dim: int, min_period: float, max_period: float) -> torch.Tensor:
    """Compute sinusoidal timestep embedding [1, dim]."""
    from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
    cpu = torch.device("cpu")
    t_tensor = torch.tensor([t], dtype=torch.float32, device=cpu)
    return create_sinusoidal_pos_embedding(t_tensor, dim, min_period, max_period, device=cpu)


def test_interface_shapes():
    """
    Verify tensor shapes match at every unit boundary.
    Uses random weights (no checkpoint loading).
    """
    print("\n" + "=" * 70)
    print("INTEGRATION SHAPE TEST  (random weights, no checkpoint)")
    print("=" * 70)

    B = BATCH_SIZE
    torch.manual_seed(0)

    # ── Unit 1: Vision encoder ───────────────────────────────────────────────
    vision_enc = NeuronVisionEncoderBlock().eval()
    pixel = torch.randn(B, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)
    with torch.no_grad():
        vision_embs = vision_enc(pixel)        # [B, 64, 960]
    assert vision_embs.shape == (B, VISION_TOKENS_PER_CAMERA, HIDDEN_SIZE), \
        f"Vision enc output: {vision_embs.shape}"
    print(f"  [1] Vision encoder:  {tuple(pixel.shape)} → {tuple(vision_embs.shape)}")

    # ── Build prefix [B, 113, 960] ───────────────────────────────────────────
    text_embs  = torch.randn(B, NUM_TEXT_TOKENS,  HIDDEN_SIZE)
    state_embs = torch.randn(B, NUM_STATE_TOKENS, HIDDEN_SIZE)
    prefix = make_prefix_embeddings(vision_embs, text_embs, state_embs)
    assert prefix.shape == (B, PREFIX_SEQ_LEN, HIDDEN_SIZE), \
        f"Prefix shape: {prefix.shape}"
    print(f"  [prefix assembled]:  text+vision+state → {tuple(prefix.shape)}")

    # ── Unit 2: VLM backbone ─────────────────────────────────────────────────
    vlm = NeuronVLMBackboneBlock().eval()
    attn_mask = torch.ones(B, PREFIX_SEQ_LEN, PREFIX_SEQ_LEN, dtype=torch.bool)
    pos_ids   = torch.arange(PREFIX_SEQ_LEN).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        conditioning = vlm(prefix, attn_mask, pos_ids)    # [B, 113, 5120]
    assert conditioning.shape == (B, NUM_CONDITIONING_TOKENS, CONDITIONING_HIDDEN_SIZE), \
        f"VLM backbone output: {conditioning.shape}"
    print(f"  [2] VLM backbone:    {tuple(prefix.shape)} → {tuple(conditioning.shape)}")

    # ── Verify conditioning contract ─────────────────────────────────────────
    from action_expert_block import NeuronSmolVLAActionHead
    head = NeuronSmolVLAActionHead()
    contract = head.get_conditioning_contract()
    contract.validate(conditioning.to(torch.bfloat16))
    print(f"  [contract validated]: num_cond_tokens={contract.num_conditioning_tokens}, "
          f"hidden={contract.conditioning_hidden_size}")

    # ── Unit 3: Action expert (single step) ──────────────────────────────────
    expert = NeuronSmolVLADenoisingWrapper().eval()
    noisy_actions = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, dtype=torch.bfloat16)
    cond_bf16     = conditioning.to(torch.bfloat16)
    t_emb         = sinusoidal_timestep_emb(1.0, TIMESTEP_EMBED_DIM, MIN_PERIOD, MAX_PERIOD)
    t_emb         = t_emb.to(torch.bfloat16).expand(B, -1)    # [B, 720]
    cross_mask    = torch.ones(B, 1, ACTION_CHUNK_SIZE, NUM_CONDITIONING_TOKENS, dtype=torch.int32)
    with torch.no_grad():
        v_t = expert(noisy_actions, cond_bf16, t_emb, cross_mask)    # [B, 50, 32]
    assert v_t.shape == (B, ACTION_CHUNK_SIZE, ACTION_DIM), \
        f"Action expert output: {v_t.shape}"
    print(f"  [3] Action expert:   {tuple(noisy_actions.shape)} → {tuple(v_t.shape)}")

    print("\n✓ ALL SHAPE CHECKS PASSED")


def test_end_to_end_denoising():
    """
    Full denoising loop: noise → 10 expert steps → predicted action chunk.
    Uses random weights; checks only shapes and no NaN/Inf.
    """
    print("\n" + "=" * 70)
    print("END-TO-END DENOISING LOOP  (10 steps, random weights)")
    print("=" * 70)

    B = BATCH_SIZE
    torch.manual_seed(1)

    # Build all three blocks
    vision_enc = NeuronVisionEncoderBlock().eval()
    vlm        = NeuronVLMBackboneBlock().eval()
    expert     = NeuronSmolVLADenoisingWrapper().eval()

    # Unit 1: vision
    pixel = torch.randn(B, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)
    with torch.no_grad():
        vision_embs = vision_enc(pixel)

    # Prefix
    text_embs  = torch.randn(B, NUM_TEXT_TOKENS,  HIDDEN_SIZE)
    state_embs = torch.randn(B, NUM_STATE_TOKENS, HIDDEN_SIZE)
    prefix     = make_prefix_embeddings(vision_embs, text_embs, state_embs)
    attn_mask  = torch.ones(B, PREFIX_SEQ_LEN, PREFIX_SEQ_LEN, dtype=torch.bool)
    pos_ids    = torch.arange(PREFIX_SEQ_LEN).unsqueeze(0).expand(B, -1)

    # Unit 2: VLM backbone
    with torch.no_grad():
        conditioning = vlm(prefix, attn_mask, pos_ids).to(torch.bfloat16)

    # Unit 3: denoising loop
    cross_mask = torch.ones(B, 1, ACTION_CHUNK_SIZE, NUM_CONDITIONING_TOKENS, dtype=torch.int32)
    x_t = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, dtype=torch.bfloat16)
    dt  = -1.0 / NUM_DENOISING_STEPS
    timesteps = [1.0 - i / NUM_DENOISING_STEPS for i in range(NUM_DENOISING_STEPS)]

    with torch.no_grad():
        for step, t in enumerate(timesteps):
            t_emb = sinusoidal_timestep_emb(t, TIMESTEP_EMBED_DIM, MIN_PERIOD, MAX_PERIOD)
            t_emb = t_emb.to(torch.bfloat16).expand(B, -1)
            v_t   = expert(x_t, conditioning, t_emb, cross_mask)
            x_t   = x_t + dt * v_t.to(x_t.dtype)

    actions = x_t    # [B, 50, 32]
    assert actions.shape == (B, ACTION_CHUNK_SIZE, ACTION_DIM)
    assert not torch.isnan(actions).any(), "NaN in output!"
    assert not torch.isinf(actions).any(), "Inf in output!"

    print(f"  Final actions: shape={tuple(actions.shape)}, "
          f"mean={actions.float().mean().item():.4f}, "
          f"std={actions.float().std().item():.4f}")
    print("\n✓ END-TO-END DENOISING LOOP PASSED (no NaN, no Inf)")


if __name__ == "__main__":
    test_interface_shapes()
    test_end_to_end_denoising()
    print("\n" + "=" * 70)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 70)
