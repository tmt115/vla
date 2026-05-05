# π0 → AWS Trainium Port

Physical Intelligence π0 (lerobot/pi0) compiled for AWS Trainium. Validated 2026-05-04.

## Hardware Requirements

| Item | Specification |
|------|---------------|
| Instance | trn1.32xlarge (16 Trainium chips, 32 NeuronCores, 512 GB HBM) |
| Neuron SDK | aws-neuronx-tools ≥ 2.18, neuronx-cc ≥ 2.18, torch_neuronx ≥ 2.9 |
| AMI | Deep Learning AMI Neuron PyTorch 2.9 |
| Python | 3.12 |

## Environment Setup

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

## Architecture Overview

```
Inputs: 3 × [1, 3, 224, 224] float32   Language: [1, 48] int64   State: [1, 6] float32
           (cameras, normalized [-1,1])   (tokenized, padded)        (robot joints)
               │                                │
               ▼                                │
┌──────────────────────────────┐               │
│  Vision Encoder (NEFF)       │               │
│  SigLIP So400m/patch14-224   │               │
│  27 layers, d=1152, h=16     │               │
│  In:  [1, 3, 224, 224]       │               │
│  Out: [1, 256, 2048] ×3      │               │
└──────────────┬───────────────┘               │
               │ ×3 cameras, scaled ×√2048      │
               │ concat with language embs →    │
               └────────────────┐              │
                                ▼              ▼
                   ┌────────────────────────────────┐
                   │  [1, 816, 2048] prefix_embs     │
                   └──────────────┬─────────────────┘
                                  ▼
                   ┌──────────────────────────────────┐
                   │  Prefix Encoder (NEFF)            │
                   │  PaliGemma Gemma 2B backbone      │
                   │  18 layers, d=2048, h=8, kv=1     │
                   │  Bidirectional attention           │
                   │  Out: [18, 2, 1, 816, 1, 256] KV  │
                   └──────────────┬───────────────────┘
                                  │ prefix_kv (passed every step)
                   ┌──────────────▼──────────────────────────┐
                   │  Suffix Denoiser × 10 Euler steps        │
                   │  (NeuronPi0DenoisingWrapper, NEFF)        │
                   │  Gemma 300M action expert                │
                   │  18 layers, d=1024, h=8, kv=1           │
                   │  Attends to prefix KV + causal suffix    │
                   │  In:  state[1,32] + x_t[1,50,32]        │
                   │       + time_emb[1,1024] + prefix_kv     │
                   │  Out: v_t[1,50,32] (Euler step)          │
                   └──────────────┬───────────────────────────┘
                                  │ x_t = x_t + dt * v_t (10×)
                                  ▼
                          actions [1, 50, 6]
```

## Compiled NEFFs

| Subgraph | NEFF Path | TP | Input Shapes | Output Shape | Compiler Flags |
|----------|-----------|----|-----------|--------------|----|
| Vision Encoder | compiled/vision_encoder/model.pt | 671 MB | 1 | [1, 3, 224, 224] float32 | [1, 256, 2048] float32 | `--auto-cast matmult --optlevel 3 --model-type unet-inference` |
| Prefix Encoder | compiled/prefix_encoder/model.pt | 3927 MB | 1 | [1, 816, 2048] bfloat16 | [18, 2, 1, 816, 1, 256] bfloat16 | `-O1 --auto-cast=none` |
| Suffix Denoiser | compiled/suffix_denoiser/model.pt | 3.6 MB + sharded weights | 1 | [1,32] + [1,50,32] + [1,1024] + [18,2,1,816,1,256] bfloat16 | [1, 50, 32] float32 | `-O1 --auto-cast=none` |

## Compile from Scratch

```bash
# Compile all (takes ~60 minutes)
python compile_all.py

# Compile individual subgraphs
python compile_all.py --only vision    # ~15 min
python compile_all.py --only prefix   # ~25 min
python compile_all.py --only suffix   # ~10 min

# Force recompile
python compile_all.py --force
```

## Run Inference

**Option A: Load pre-compiled NEFFs**
```python
from run_inference import load_model, generate_actions
import torch

model = load_model()  # loads from compiled/ and weights/

images = [torch.randn(1, 3, 224, 224) * 0.5 for _ in range(3)]  # 3 cameras
lang_tokens = torch.zeros(1, 48, dtype=torch.long)
lang_masks = torch.ones(1, 48, dtype=torch.bool)
state = torch.zeros(1, 6)

actions = generate_actions(model, images, lang_tokens, lang_masks, state)
# actions: numpy array [50, 6] — 50 action steps × 6 joints
```

**Option B: Compile then run**
```bash
python compile_all.py
python run_inference.py
```

## Benchmark

```bash
python benchmark.py
```

Results on trn1.32xlarge, TP=1, batch=1, 50 iterations (20 warmup discarded):

| Subgraph | Mean | P95 | Throughput |
|----------|------|-----|------------|
| Vision Encoder (1 camera) | 5.8 ms | 5.8 ms | 173 inf/sec |
| Vision Encoder (×3 cameras) | 17.3 ms | — | — |
| Prefix Encoder (Gemma 2B, 816 tokens) | 168.8 ms | 168.9 ms | 5.9 inf/sec |
| Suffix Denoiser (Gemma 300M, ×10 steps) | ~55 ms | — | — |
| **End-to-end pipeline** | **241.3 ms** | **241.7 ms** | **4.14 policy steps/sec** |

## Validate Correctness

```bash
python validate_neffs.py
```

Results (trn1.32xlarge, 2026-05-05):

| Subgraph | mean_diff | cos_sim | Threshold | Result |
|----------|-----------|---------|-----------|--------|
| Vision Encoder | 0.0017 | 1.000090 | atol=0.1, cos>0.999 | ✅ PASSED |
| Prefix Encoder (K) | 0.0102 | 1.000266 | atol=0.1, cos>0.997 | ✅ PASSED |
| Prefix Encoder (V) | 0.0149 | 1.000131 | atol=0.1, cos>0.997 | ✅ PASSED |
| Suffix Denoiser | 0.0119 | 0.999986 | atol=0.15, cos>0.999 | ✅ PASSED |

## Compiler Flags Rationale

| Subgraph | Flag | Reason |
|----------|------|--------|
| Vision | `--model-type unet-inference` | +6% over transformer for ViT/SigLIP; correct for non-causal encoders |
| Vision | `--auto-cast matmult` | BF16 matmuls, FP32 accumulators — 50% NEFF size reduction |
| Prefix | `-O1 --auto-cast=none` | Bidirectional prefix attention ≠ causal LM; `--model-type=transformer` unsafe here |
| Suffix | `-O1 --auto-cast=none` | Flow-matching denoiser — `--model-type=transformer` causes cos_sim=0.916 on DiT-style models |

## TP Design Rationale

All subgraphs at TP=1:
- **Vision encoder**: 400M params, ViT structure — single NeuronCore sufficient. TP>1 adds sync overhead without benefit at batch=1.
- **Prefix encoder**: 2B params, runs once per inference call (not the bottleneck). TP=8 would require NxDI ModelBuilder integration and is left for future optimization.
- **Suffix denoiser**: 300M params, called 10× — NeuronDenoisingWrapper/ModelBuilder at TP=1. TP=8 is the path for latency-critical deployments.

## Known Deviations

| Deviation | Impact | Workaround |
|-----------|--------|-----------|
| DynamicCache bypassed — static KV tensor instead | Zero numerical impact | Manual layer iteration in prefix/suffix encoders; KV stored as `[18, 2, B, 816, 1, 256]` |
| `create_causal_mask` bypassed — static all-attend mask for prefix | Zero impact (prefix is bidirectional anyway) | Pre-computed `zeros` mask as register_buffer |
| Sinusoidal embedding float64 → float32 | Negligible (period vector precision) | Period vector pre-computed as float32 buffer |
| `_gated_residual` gate=None path inlined as `x + y` | Zero impact (use_adarms=False for π0) | Static branch elimination at trace time |

## Directory Structure

```
pi0-port/
├── config_constants.py      # All architecture constants (from checkpoint)
├── vision_encoder.py        # SigLIP + projector, NeuronVisionEncoder
├── prefix_encoder.py        # Gemma 2B backbone, NeuronPrefixEncoder
├── suffix_denoiser.py       # Gemma 300M denoiser, NeuronPi0DenoisingWrapper
├── pi0_neuron.py            # Full pipeline assembly, Pi0NeuronModel
├── compile_all.py           # Compile all 3 NEFFs
├── validate_neffs.py        # NEFF correctness validation
├── run_inference.py         # Main inference script + API
├── benchmark.py             # Latency benchmark
├── STATUS.md                # Phase tracker
├── weights/                 # lerobot/pi0 checkpoint (14 GB)
│   ├── model.safetensors
│   └── config.json
├── compiled/
│   ├── vision_encoder/      # NEFF + weights
│   ├── prefix_encoder/      # NEFF + weights
│   └── suffix_denoiser/     # NEFF + weights
├── tests/
│   ├── test_vision_encoder.py
│   ├── test_prefix_encoder.py
│   ├── test_suffix_denoiser.py
│   ├── test_integration_cpu.py
│   └── test_weight_mapping.py
└── skills/
    ├── SKILL.md
    └── reference/
```
