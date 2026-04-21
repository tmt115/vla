# SmolVLA → AWS Trainium NxDI Port Notes

## Compilation units

| Unit | File | TP degree | Input | Output |
|------|------|-----------|-------|--------|
| Vision encoder | vision_encoder_block.py | 1 | [B,3,512,512] BF16 | [B,64,960] BF16 |
| VLM backbone | vlm_backbone_block.py | 8 | [B,113,960] BF16 | [B,113,5120] BF16 |
| Action expert | action_expert_block.py | 2 | [B,50,32],[B,113,5120],[B,720],[B,1,50,113] | [B,50,32] BF16 |

## num_conditioning_tokens

`PREFIX_SEQ_LEN = 48 + 64 + 1 = 113`

- 48 text tokens (tokenizer_max_length)
- 64 vision tokens per camera (1024 SigLIP patches / 4² pixel-shuffle)
- 1 state token (robot proprioception → Linear(32→960) → 1 token)

Scales with number of cameras: each additional camera adds 64 more vision tokens.

## Vision bucket sizes

Only one bucket implemented (1 camera = 64 vision tokens → PREFIX_SEQ_LEN = 113).
For multi-camera, compile separate NEFFs with different PREFIX_SEQ_LEN values:
- 1 cam: 48 + 64 + 1 = 113
- 2 cam: 48 + 128 + 1 = 177
- 3 cam: 48 + 192 + 1 = 241

Both the VLM backbone and the action expert attention_mask must use the matching bucket size.

## TP degree per subgraph

| Subgraph | TP degree | Rationale |
|----------|-----------|-----------|
| Vision encoder | 1 | ~86M params; tiny; no need to shard |
| VLM backbone | 8 | ~500M params; Q/K/V/O proj rows dividible by 8; hidden=960 (64×15) |
| Action expert | 2 | ~100M params; Q proj [720→960], O proj [960→720], intermediate=2048 |

Verify TP degrees at compile time: `torch_neuronx.trace()` will fail if hidden dims
are not evenly divisible by the TP degree.

## Compiler flags

Recommended for all three units:
```
--enable-saturate-infinity
--model-type transformer
```

For VLM backbone (GQA with 15/5 heads):
```
--enable-mixed-precision-accumulation
```

For action expert (interleaved self/cross-attn):
```
--enable-mixed-precision-accumulation
--internal-hlo-passes merge-consecutive-transposes
```

## L/2 exit — VLM layer truncation

The task brief's "L/2 exit" refers to the VLM using 16 of the 32 base SmolLM2 layers.
This truncation is **pre-baked in the checkpoint** — the safetensors file contains exactly
16 decoder layers under `model.vlm_with_expert.vlm.model.text_model.layers.{0..15}`.
No runtime truncation is needed; just load all 16 layers.

Code: `SmolVLMWithExpertModel(..., num_vlm_layers=16)` sets this at init time.

## Denoising loop design

The loop is intentionally **not compiled** into the NEFF:
- `NeuronSmolVLAActionHead.generate_actions()` runs N iterations on CPU
- Each iteration calls the compiled `NeuronSmolVLADenoisingWrapper` once
- Timestep embeddings (sinusoidal, 720-dim) are computed on CPU before each call
- Only the single-step forward (attention over 50 action tokens) is compiled

Timestep sequence: `t = [1.0, 0.9, ..., 0.1]` (flow-matching, 10 steps default)
Step update: `x_{t-dt} = x_t + dt * v_t` where `dt = -1/num_steps`

## Key architectural deviations from base SmolLM2

| Feature | Base SmolLM2 | SmolVLA VLM | SmolVLA Expert |
|---------|-------------|-------------|----------------|
| RoPE max_wavelength | 10000 | 10000 | 10000 |
| Attention | GQA 15/5 | GQA 15/5 | GQA 15/5 (self) + cross |
| Layer type | Uniform decoder | Uniform decoder | Interleaved self/cross |
| Hidden size | 960 | 960 | 720 (0.75×) |
| Intermediate | 2560 | 2560 | 2048 |
| Cross-attn K/V in | N/A | N/A | 320 (from VLM KV) |

**Critical**: `apply_rope` uses `max_wavelength=10_000` (hardcoded in
`smolvlm_with_expert.py`), NOT the `rope_theta=100_000` from the HF config.
This affects both the VLM and the action expert Q/K.

## Conditioning tensor layout

The VLM backbone packs its KV cache into `[B, 113, 5120]`:
```
conditioning[b, t, :] = concat over 8 cross-attn layers:
    [K_l0_t, V_l0_t, K_l1_t, V_l1_t, ..., K_l7_t, V_l7_t]
```
where `K_li` and `V_li` are each 320-dim (5 KV heads × 64 head_dim).

Cross-attn layers are VLM odd layers: {1, 3, 5, 7, 9, 11, 13, 15} (0-indexed).
Expert cross-attn layer index `c = layer_idx // 2`.

To unpack inside the denoiser:
```python
cond = conditioning.reshape(B, 113, 8, 2, 320)
k_vlm = cond[:, :, c, 0, :]  # [B, 113, 320]
v_vlm = cond[:, :, c, 1, :]  # [B, 113, 320]
```

## Test results (CPU parity)

| Test | Max abs diff | Tolerance | Status |
|------|-------------|-----------|--------|
| Vision encoder (random weights, BF16) | 9.77e-04 | 1e-2 | PASS |
| VLM backbone (random weights, BF16) | 4.88e-04 | 5e-2 | PASS |
| Action expert single step (random, BF16) | 0.00e+00 | 1e-3 | PASS |
| Action expert 10-step loop (random, BF16) | 0.00e+00 | 1e-2 | PASS |

Action expert shows zero diff because the reference and neuron blocks are identical
architectures run on CPU with no approximation.
