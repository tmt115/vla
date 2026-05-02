# Port Status — GR00T N1.7 → AWS Trainium

## Phase 0 — Complete
- Environment verified: 2026-04-26
  - torch 2.9.1, torch_neuronx OK, NxDI OK
  - diffusers 0.35.2, transformers 4.57.6
- Checkpoint inspected: nvidia/GR00T-N1.7-3B (1031 tensors, 2 safetensors shards)
- Isaac-GR00T cloned: /home/ubuntu/Isaac-GR00T
- config_constants.py written: yes (all values cross-checked against checkpoint tensor shapes)
- Test harness: tests/block_testing_utils.py present

## Phase 1 — Complete
- Root cause identified: layers constructed in __init__ before parallel_state active
- All wrapper classes redesigned with lazy init pattern

## Phase 2 — Complete (2026-04-27)

### Block 1: Backbone (vlm_backbone_block.py) — REWRITTEN
- GR00TBackboneLLM: uses NeuronQwen3VLDecoderLayer (NxDI), 16 layers
- NeuronGR00TBackboneWrapper: lazy init in load_module(), auto-init in forward()
- load_module() skips weight loading on Trainium (uses sharding instead)
- CPU test: forward OK (shape [1,256,2048], no NaN), self-consistent

### Block 2: VL Self-Attention (vl_self_attention_block.py) — REWRITTEN
- GR00TVLSelfAttnModel: 4-layer, ColumnParallelLinear/RowParallelLinear with bias=True
- FFN: GELU tanh, bias on both projections (NOT NeuronLlamaMLP which drops bias)
- CPU test: cos_sim=0.999952 vs Isaac-GR00T SelfAttentionTransformer

### Block 3: DiT (dit_block.py) — FIXED
- NeuronGR00TDenoisingWrapper: model=None in __init__, constructed in load_module()
- load_module() skips weight loading on Trainium (uses sharding instead)
- CPU test: forward OK (shape [1,41,1024], no NaN)

## Phase 3 — Complete (2026-04-27)
- compile_all.py: rewritten, all three blocks compile correctly
- CPU pre-compile validate: all three blocks PASS
- _VLSelfAttnHead: thin NeuronActionHeadBase subclass for VL self-attn

## Phase 4 — Complete (2026-04-27)
- Weight mapping verified for all three blocks
- Backbone: 177 keys, shapes and values match checkpoint
- VL self-attn: 64 keys, shapes match
- DiT: 452 keys, shapes match

## Phase 5 — Complete (2026-05-02, revalidated with -O1 flags)

### Compiler Flags Audit (2026-05-02)
All three TP=8 blocks recompiled with `-O1` only (removed `--model-type=transformer`
and `--tensorizer-options`). Rationale: `--model-type=transformer` activates NxDI's
flash-attention kernel (attn_kernel_enabled) which produces ~3% per-layer error on
trn1, compounding to cos_sim=0.547 over 16 backbone layers. With `-O1`, the standard
XLA matmul+softmax path is used, giving exact bfloat16 results.

### TP Verification
All three TP=8 blocks verified via ColumnParallelLinear shard shapes:
- Backbone q_proj: [256, 2048] = 2048/8 ✅
- Backbone gate_proj: [768, 2048] = 6144/8 ✅
- VL self-attn to_q: [256, 2048] = 2048/8 ✅
- VL self-attn ff_up: [1024, 2048] = 8192/8 ✅
- DiT to_q: [192, 1536] = 1536/8 ✅

### NEFF Correctness Validation (2026-05-02, validate_neffs.py, -O1 flags)
| Block | cos_sim | mean_diff | Threshold | Status |
|-------|---------|-----------|-----------|--------|
| vl_self_attn | 0.999941 | 0.006666 | cos>0.999, atol<0.05 | PASS |
| dit | 0.999930 | 0.001530 | cos>0.950, atol<0.50 | PASS |
| backbone | 0.999661 | 0.042674 | cos>0.997, atol<0.10 | PASS |

Note: Previous backbone cos_sim=0.547 was due to --model-type=transformer activating
flash-attention on trn1. Fixed by switching to -O1 only. All three now PASS.

### ViT NEFF Compilation (2026-05-02)
- Dynamic op: Qwen3VLVisionAttention uses `torch.split(tensor, lengths.tolist())`
  with cu_seqlens — data-dependent, incompatible with neuronx-cc as-is.
- Fix: Monkey-patched all 24 ViT blocks' attention forward() to use standard
  scaled dot-product attention (numerically identical for single-image, no split needed).
- Also replaced rot_pos_emb and fast_pos_embed_interpolate with pre-computed static buffers.
- Compiler flags: `--auto-cast=matmult --optlevel 3 --model-type=unet-inference`
- NEFF size: 536 MB (TP=1, single device)
- ViT NEFF vs CPU cos_sim: 0.998119 ✓

## Phase 6 — Complete (2026-05-02)

### Compiler Flags: [-O1] for all subgraphs (--model-type=transformer removed after audit)

### TP Verification: PASSED per subgraph (shard shapes confirmed)

### NEFFs compiled: backbone, vl_self_attn, dit, vit
| NEFF | Size | TP | cos_sim |
|------|------|----|---------|
| backbone | 1.4MB + 8x shards | 8 | 0.999661 |
| vl_self_attn | 564KB + 8x shards | 8 | 0.999941 |
| dit | 4.3MB + 8x shards | 8 | 0.999930 |
| vit | 536MB | 1 | 0.998119 |

### ISA Kernel Settings (trn1)
- attn_kernel_enabled=False explicitly set in backbone config ✓
- Other kernel flags default to False (qkv_kernel_enabled, mlp_kernel_enabled, etc.)
- context_encoding_buckets=None (static seq_len, no bucketing needed)
- NKI kernels are trn2+ only; all correctly disabled for trn1

### Open-loop MSE vs HF (2026-05-02, 3 trajectories, dummy inputs)
| Trajectory | MSE | cos_sim |
|-----------|-----|---------|
| 1 | 0.000063 | 0.999968 |
| 2 | 0.000063 | 0.999969 |
| 3 | 0.000059 | 0.999970 |
| **Mean** | **0.000062** | **0.999969** |

### Per-Subgraph Latency Breakdown (trn1.32xlarge, TP=8, batch=1, 4 denoising steps)
| Subgraph | Mean | P95 |
|----------|------|-----|
| ViT (CPU) | 344.80 ms | 415.66 ms |
| ViT (NEFF) | 4.94 ms | 4.97 ms |
| Backbone NEFF | 6.64 ms | 6.93 ms |
| VL self-attn NEFF | 2.31 ms | 2.50 ms |
| DiT NEFF (per step) | 6.01 ms | 6.32 ms |
| **Total excl. ViT** | **43.96 ms** | 47.88 ms |
| Total incl. ViT CPU | 610.80 ms | 706.14 ms |

Throughput: 22.7 inferences/sec (excl. ViT)

### Final Benchmark (trn1.32xlarge, TP=8, batch=1, 4 denoising steps, 30 iter)
- Mean: 43.96ms | P95: 47.88ms | Throughput: 22.7 inferences/sec
- Per-step DiT latency: 6.01ms

### Documentation
- README.md: written (architecture, setup, compile, validate, benchmark, deviations)
- groot_n1_trainium.ipynb: written (8 cells covering setup through validation)
- run_inference.py: verified + profile=True timing added to generate_actions()
- benchmark.py: verified
- benchmark_subgraph.py: written (per-subgraph timing, 30 iters, 10 warmup)
- open_loop_eval.py: written (3 trajectories, MSE/cos_sim vs HF)

### Known Deviations
1. DiT: all even blocks attend ALL 256 conditioning tokens (no image/text mask split).
   cos_sim=0.9999 — negligible.
2. ViT static attention patch: cu_seqlens split replaced with full-sequence attention.
   For single image, numerically identical. cos_sim=0.998119.
3. ViT deepstack outputs dropped in NEFF (unused in GR00T inference pipeline).
