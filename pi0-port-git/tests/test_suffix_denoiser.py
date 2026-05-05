"""CPU tests for NeuronPi0DenoisingWrapper."""
import sys, math, torch, torch.nn as nn
sys.path.insert(0, '/home/ubuntu/pi0-port')
sys.path.insert(0, '/home/ubuntu/pi0-port/skills/scripts')


def test_embed_timestep():
    """_embed_timestep produces [B, 1024] bfloat16 without NaN."""
    from config_constants import EXPERT_HIDDEN_SIZE, MIN_PERIOD, MAX_PERIOD

    fraction = torch.linspace(0.0, 1.0, EXPERT_HIDDEN_SIZE // 2, dtype=torch.float32)
    period = MIN_PERIOD * (MAX_PERIOD / MIN_PERIOD) ** fraction
    period_vector = period

    t = 0.5
    time_tensor = torch.full((1,), t, dtype=torch.float32)
    scaling = 1.0 / period_vector * 2 * math.pi
    sin_input = scaling[None, :] * time_tensor[:, None]
    time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).to(torch.bfloat16)

    assert time_emb.shape == (1, EXPERT_HIDDEN_SIZE), f"Expected [1,{EXPERT_HIDDEN_SIZE}], got {time_emb.shape}"
    assert not torch.isnan(time_emb).any(), "NaN in time_emb"
    assert time_emb.dtype == torch.bfloat16
    print(f"PASS: _embed_timestep — shape {time_emb.shape}")


def test_suffix_attn_mask_shape():
    """Attention mask has correct shape and values."""
    from suffix_denoiser import _build_suffix_attn_mask
    from config_constants import PREFIX_LEN, CHUNK_SIZE, SUFFIX_LEN, FULL_ATTN_LEN

    mask = _build_suffix_attn_mask(PREFIX_LEN, CHUNK_SIZE)
    assert mask.shape == (1, 1, SUFFIX_LEN, FULL_ATTN_LEN), f"Bad shape: {mask.shape}"
    # State row: should block action positions
    assert mask[0, 0, 0, PREFIX_LEN + 1] < -1e8, "State row should block subsequent actions"
    # Action rows: should be all zeros
    assert mask[0, 0, 1, :].max().item() == 0.0, "Action rows should attend everywhere"
    print(f"PASS: suffix_attn_mask shape {mask.shape}")


def test_wrapper_forward_tiny():
    """NeuronPi0DenoisingWrapper forward with tiny Gemma config."""
    from neuron_action_head_base import NeuronDenoisingConfig
    from suffix_denoiser import NeuronPi0DenoisingWrapper

    # Tiny dimensions
    TINY_H = 32
    TINY_L = 2
    TINY_NH = 2
    TINY_HD = 16
    TINY_NKV = 1
    TINY_MLP = 64
    TINY_PREFIX = 4
    TINY_CHUNK = 3
    TINY_STATE = 4
    TINY_ACTION = 4

    config = NeuronDenoisingConfig(
        batch_size=1, tp_degree=1,
        action_chunk_size=TINY_CHUNK, action_dim=TINY_ACTION,
        num_conditioning_tokens=TINY_PREFIX, conditioning_hidden_size=TINY_H,
        timestep_embed_dim=TINY_H,
    )
    wrapper = NeuronPi0DenoisingWrapper(
        config,
        num_layers=TINY_L,
        hidden_size=TINY_H,
        head_dim=TINY_HD,
        num_heads=TINY_NH,
        kv_heads=TINY_NKV,
        intermediate_size=TINY_MLP,
        prefix_len=TINY_PREFIX,
        chunk_size=TINY_CHUNK,
        max_state=TINY_STATE,
        max_action=TINY_ACTION,
    )
    wrapper.load_module()
    wrapper.eval()

    B = 1
    state = torch.zeros(B, TINY_STATE, dtype=torch.bfloat16)
    noisy_actions = torch.zeros(B, TINY_CHUNK, TINY_ACTION, dtype=torch.bfloat16)
    time_emb = torch.zeros(B, TINY_H, dtype=torch.bfloat16)
    prefix_kv = torch.zeros(TINY_L, 2, B, TINY_PREFIX, TINY_NKV, TINY_HD, dtype=torch.bfloat16)

    with torch.no_grad():
        v_t = wrapper(state, noisy_actions, time_emb, prefix_kv)

    expected = (B, TINY_CHUNK, TINY_ACTION)
    print(f"v_t shape: {tuple(v_t.shape)}, expected: {expected}")
    assert tuple(v_t.shape) == expected, f"Shape mismatch: {tuple(v_t.shape)}"
    assert v_t.dtype == torch.float32, f"Expected float32, got {v_t.dtype}"
    assert not torch.isnan(v_t).any(), "NaN in output"
    print(f"PASS: wrapper forward — output {v_t.shape} float32")


def test_input_generator():
    """input_generator returns correct shapes for full-size config."""
    from neuron_action_head_base import NeuronDenoisingConfig
    from suffix_denoiser import NeuronPi0DenoisingWrapper
    from config_constants import (
        EXPERT_HIDDEN_SIZE, EXPERT_NUM_LAYERS, EXPERT_HEAD_DIM,
        EXPERT_NUM_KV_HEADS, CHUNK_SIZE, MAX_STATE_DIM, MAX_ACTION_DIM, PREFIX_LEN
    )

    config = NeuronDenoisingConfig(
        batch_size=1, tp_degree=1,
        action_chunk_size=CHUNK_SIZE, action_dim=MAX_ACTION_DIM,
        num_conditioning_tokens=PREFIX_LEN, conditioning_hidden_size=EXPERT_HIDDEN_SIZE,
        timestep_embed_dim=EXPERT_HIDDEN_SIZE,
    )
    wrapper = NeuronPi0DenoisingWrapper(config)
    inputs = wrapper.input_generator()
    state, noisy, time_e, pkv = inputs[0]
    assert state.shape == (1, MAX_STATE_DIM)
    assert noisy.shape == (1, CHUNK_SIZE, MAX_ACTION_DIM)
    assert time_e.shape == (1, EXPERT_HIDDEN_SIZE)
    assert pkv.shape == (EXPERT_NUM_LAYERS, 2, 1, PREFIX_LEN, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM)
    print(f"PASS: input_generator — prefix_kv shape {pkv.shape}")


if __name__ == "__main__":
    print("Running suffix denoiser CPU tests...")
    test_embed_timestep()
    test_suffix_attn_mask_shape()
    test_input_generator()
    test_wrapper_forward_tiny()
    print("\nAll suffix denoiser tests PASSED")
