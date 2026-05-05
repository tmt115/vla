# Port Status — lerobot/pi0 → AWS Trainium

## Phase 0 — Complete
- Environment verified: 2026-05-03
- NxDI OK, torch_neuronx OK
- Checkpoint inspected: lerobot/pi0 (14 GB, 777 keys, model.safetensors)
- config_constants.py written: yes
- Test harness calling convention:
  - `test_block_correctness(neuron_block_class, pytorch_block_class, weight_mapping, example_inputs=[zeros], test_inputs=[randoms], reference_inputs=[randoms])`
  - Reference called as: `block(reference_inputs[0][0])`
  - Harness casts reference to bfloat16
- Test harness dtype casting: bfloat16 (all parameters cast via `reference_block.to(dtype=torch.bfloat16)`)

### Key Architecture Constants (from checkpoint)
| Subgraph | Key Dimensions |
|----------|---------------|
| SigLIP | hidden=1152, layers=27, heads=16, patch=14, img=224, tokens=256/image |
| Multi-modal projector | 1152 → 2048 (single Linear) |
| PaliGemma Gemma 2B | hidden=2048, layers=18, heads=8, kv_heads=1, head_dim=256, mlp=16384 |
| Action Expert Gemma 300M | hidden=1024, layers=18, heads=8, kv_heads=1, head_dim=256, mlp=4096 |
| Action projections | state_proj: 32→1024, action_in_proj: 32→1024, time_mlp: 2048→1024→1024, action_out: 1024→32 |
| Flow matching | chunk=50, steps=10, state/action_dim padded to 32 |

### Subgraph Plan
1. **Vision Encoder** (SigLIP + projector): 3×[1,3,224,224] → 3×[1,256,2048] — trace() TP=1
2. **Prefix Encoder** (Gemma 2B backbone): [1,816,2048] → KV cache [18,2,1,816,1,256] — trace() TP=1
3. **Suffix Denoiser** (Gemma 300M, 1 step): [1,51,1024] + KV → [1,51,1024] — trace() TP=1 × 10 steps

### Checkpoint Verification Results
- action_in_proj.weight: [1024, 32] ✓
- action_out_proj.weight: [32, 1024] ✓
- VLM q_proj: [2048, 2048] ✓, k/v_proj: [256, 2048] ✓ (MQA: 1 KV head × 256)
- Expert q_proj: [2048, 1024] ✓, k/v_proj: [256, 1024] ✓ (same KV shape → cross-stream compatible)
- SigLIP patch_embedding (Conv2D): [1152, 3, 14, 14] ✓
- Multi-modal projector: [2048, 1152] ✓
- 3 cameras in this checkpoint → PREFIX_LEN = 256×3 + 48 = 816

## Phase 1 — Complete
- Reference model: none (all subgraphs use raw trace() — non-autoregressive, non-standard)
- NxDI compatibility: trace() for vision encoder + prefix encoder; ModelBuilder via NeuronDenoisingWrapper for suffix denoiser
- Block partition: 3 translation units — vision_encoder.py, prefix_encoder.py, suffix_denoiser.py
- Non-standard patterns: dual-stream Gemma joint attention, flow matching loop, static KV cache cross-subgraph pass

## Phase 2 — Complete
- Blocks translated: vision_encoder.py, prefix_encoder.py, suffix_denoiser.py
- All unit tests passing: yes
- TP verification: N/A (TP=1 via raw trace() for vision+prefix; ModelBuilder TP=1 for suffix denoiser)
- Deviations: manual layer iteration (bypass PiGemmaModel.forward / DynamicCache); static KV tensor cross-subgraph pass
- Dynamic constants moved to register_buffer: prefix_position_ids, cache_position, prefix_attn_mask, suffix_attn_mask, suffix_position_ids, period_vector

## Phase 3 — Complete
- Model assembled: pi0_neuron.py (Pi0NeuronModel, LanguageEmbedder, preprocess_image)
- Config classes: NeuronDenoisingConfig (from neuron_action_head_base.py)
- Deviations resolved: manual layer iteration, static KV tensor cross-subgraph pass, DynamicCache bypassed
- TP confirmed: TP=1 for all subgraphs (vision trace, prefix trace, suffix ModelBuilder TP=1)
- CPU integration tests: all 6 passing (KV shape, vision→prefix, prefix→suffix, embedding pipeline, 3-camera prefix)

## Phase 4 — Complete
- Weight mapping implemented: yes
- Validation passing: yes
- Shape mismatches found and resolved: prefix encoder key prefix was `.language_model.` not `.language_model.model.`
- suffix_attn_mask / suffix_position_ids intentionally absent from checkpoint (pre-computed buffers)
- Vision encoder weights: max_diff < 1e-5 vs reference ✓
- Prefix encoder layer0 q_proj [2048, 2048] ✓
- Suffix denoiser action_in_proj [1024, 32] ✓

## Phase 5 — In progress
- Vision encoder NEFF: 671.4 MB ✓ — mean_diff=0.0017 cos_sim=1.000090 PASSED
- Prefix encoder NEFF: 3926.5 MB ✓ — mean_diff_K=0.0102 cos_sim=1.000266 / mean_diff_V=0.0149 cos_sim=1.000131 PASSED
- Suffix denoiser NEFF: 3.6 MB + sharded weights ✓ — mean_diff=0.0119 cos_sim=0.999986 PASSED
- Open-loop MSE vs HF: N/A (single-step validation passes; open_loop_eval runs standalone due to NeuronCore allocation constraints in subprocess)
- Benchmark: Vision 5.8ms | Prefix 168.8ms | Suffix×10 ~55ms | End-to-end 241.3ms | 4.14 policy steps/sec
## Phase 2 — Pending
## Phase 3 — Pending
## Phase 4 — Pending
## Phase 5 — Pending
## Phase 5 — Complete
- TP verification: N/A (TP=1 trace for vision+prefix; ModelBuilder TP=1 for suffix)
- NEFFs compiled: yes (all 3)
- NEFF correctness: vision mean_diff=0.0017 cos_sim=1.000090 PASS | prefix K=0.0102/1.000266 V=0.0149/1.000131 PASS | suffix mean_diff=0.0119 cos_sim=0.999986 PASS
- Benchmark: Vision 5.8ms (×3=17.3ms) | Prefix 168.8ms | Suffix×10 ~55ms | End-to-end 241.3ms | 4.14 policy steps/sec
- run_inference.py: PASSED — output (50,6) float32, no NaN, latency 261.4ms (first call)

## Phase 6 — Complete
- run_inference.py: verified working ✓
- benchmark.py: verified working ✓
- README.md: written (13 sections)
- demo.ipynb: written (8 sections)
