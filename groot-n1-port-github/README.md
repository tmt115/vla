# GR00T N1.7-3B → AWS Trainium Port

Port of NVIDIA GR00T N1.7-3B (vision-language-action robot model) to AWS Trainium using NxDI (NeuronX Distributed Inference) with TP=8.

Validated 2026-05-02. All four subgraphs (ViT, backbone, VL self-attn, DiT) compiled and validated.

## Hardware Requirements

- AWS `trn1.32xlarge` (32 NeuronCores, 512 GB Neuron memory)
- ~600 GB disk for checkpoint + compiled NEFFs + weight shards
- ~64 GB CPU RAM for weight loading

## Environment Setup

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd /home/ubuntu/groot-n1-port
```

Pre-installed packages in venv:
- `torch 2.9.1` + `torch_neuronx` (NeuronSDK 2.9)
- `neuronx_distributed_inference` (NxDI 2.28)
- `transformers 4.57.6`, `diffusers 0.35.2`, `safetensors`

## Architecture Overview

GR00T N1.7-3B has four subgraphs. All are compiled to Neuron NEFFs:

```
Image (256x256)
    │
    ▼
[ViT NEFF]                   Qwen3-VL ViT (24 layers, 1024-dim)
    │ [64, 2048] merged tokens
    ▼
[Backbone NEFF] TP=8         16-layer Qwen3-VL-2B LLM (text+image fusion)
    │ [1, 256, 2048]
    ▼
[VLLN + VL self-attn NEFF] TP=8    4-layer transformer (refine conditioning)
    │ [1, 256, 2048] conditioning tokens
    ▼
╔══════════════════════════════════════╗
║  Flow-matching denoising loop (4x)   ║
║  DiT NEFF TP=8                       ║
║  [noisy_actions] → [velocity_pred]   ║
╚══════════════════════════════════════╝
    │
    ▼
State/Action decoder (CPU)
    │
    ▼
Actions [1, 40, 132] BF16
```

### CPU components (not compiled to NEFF)
- State/action encoder/decoder (CategorySpecificMLP, per-embodiment weights)
- TimestepEncoder (sinusoidal + 2-layer MLP)
- Positional embeddings
- VLLN (LayerNorm before VL self-attn)
- Noise schedule (linear flow matching)

## Compiled NEFFs

| NEFF | Size | TP | Input | Output |
|------|------|----|-------|--------|
| ViT | 536 MB | 1 | [256, 1536] BF16 | [64, 2048] BF16 |
| Backbone | 1.4 MB + 8x shards | 8 | [1,256,2048] + cos/sin | [1,256,2048] |
| VL self-attn | 564 KB + 8x shards | 8 | [1,256,2048] | [1,256,2048] |
| DiT | 4.3 MB + 8x shards | 8 | sa_embs + cond + temb + mask | [1,41,1024] |

## Compile from Scratch

```bash
# All three TP=8 NEFFs (backbone, vl_self_attn, dit)
python compile_all.py 2>&1 | tee compile.log

# ViT NEFF (TP=1, single-device)
python compile_vit.py 2>&1 | tee compile_vit.log

# Individual blocks
python compile_all.py --only backbone
python compile_all.py --only vl_self_attn
python compile_all.py --only dit

# Force recompile
python compile_all.py --force
```

Compilation times on trn1.32xlarge (warm neuronx-cc cache):
- backbone: ~25 s
- vl_self_attn: ~10 s
- dit: ~100 s
- vit: ~120 s

First-time compilation (cold cache): ~2 hours total.

## Run Inference

```bash
python run_inference.py
```

Runs smoke test with dummy inputs. Output: `[1, 40, 132]` BF16 actions.

### Python API

```python
from run_inference import load_model, GR00TModel
from vlm_backbone_block import run_vit_cpu, make_backbone_inputs
import torch

model = load_model()   # loads all NEFFs + CPU weights
hf_sd = model._hf_sd  # full checkpoint (needed for ViT weights)

# Option A: Use pre-embedded inputs (if you already ran ViT)
backbone_in = make_backbone_inputs(B=1, n_text=16)
inputs_embeds, pre_cos, pre_sin = run_vit_cpu(
    backbone_in['input_ids'], backbone_in['attention_mask'],
    backbone_in['pixel_values'], backbone_in['image_grid_thw'],
    hf_sd=hf_sd,
)

state = torch.zeros(1, 1, 132, dtype=torch.bfloat16)
embodiment_id = torch.zeros(1, dtype=torch.long)

actions = model.generate_actions(
    inputs_embeds=inputs_embeds,
    pre_cos=pre_cos,
    pre_sin=pre_sin,
    state=state,
    embodiment_id=embodiment_id,
    num_steps=4,
)
# actions: [1, 40, 132] BF16

# Option B: Use compiled ViT NEFF directly
import torch, torch.jit
vit_neff = torch.jit.load('compiled/vit/model.pt')
# pixel_values: [256, 3*2*16*16=1536] BF16
vit_out = vit_neff(pixel_values)  # [64, 2048]
```

## Benchmark

```bash
# Full end-to-end (70 iters, 20 warmup)
python benchmark.py

# Per-subgraph breakdown (30 iters, 10 warmup)
python benchmark_subgraph.py
```

### Performance on trn1.32xlarge (TP=8, batch=1, 4 denoising steps)

| Subgraph | Mean latency | P95 |
|----------|-------------|-----|
| ViT (CPU) | 344.80 ms | 415.66 ms |
| ViT (NEFF) | **4.94 ms** | 4.97 ms |
| Backbone NEFF | 6.64 ms | 6.93 ms |
| VL self-attn NEFF | 2.31 ms | 2.50 ms |
| DiT (per step) | 6.01 ms | 6.32 ms |
| **Total excl. ViT** | **43.96 ms** | 47.88 ms |
| Total incl. ViT CPU | 610.80 ms | 706.14 ms |
| Total incl. ViT NEFF | ~55 ms (estimated) | |

**Throughput**: 22.7 inferences/sec (excl. ViT preprocessing)

## Validate Correctness

```bash
python validate_neffs.py
```

### Validated cos_sim (2026-05-02, -O1 flags)

| Subgraph | cos_sim | mean_diff | Threshold | Status |
|----------|---------|-----------|-----------|--------|
| VL self-attn | 0.999941 | 0.00667 | >0.999 | PASS |
| DiT | 0.999930 | 0.00153 | >0.950 | PASS |
| Backbone | 0.999661 | 0.04267 | >0.997 | PASS |
| ViT NEFF | 0.998119 | - | >0.990 | PASS |

### Open-loop eval (Neuron vs HF, 3 trajectories)

```bash
python open_loop_eval.py
```

| Metric | Value |
|--------|-------|
| Mean MSE | 0.000062 |
| Mean cos_sim | 0.999969 |

## Compiler Flags

All four NEFFs use `-O1` only (no `--model-type=transformer`):

```python
def get_compiler_args(self):
    return "-O1"
```

**Why -O1 (no --model-type=transformer):**
- `--model-type=transformer` activates causal-LM-specific XLA fusions that are
  inappropriate for the backbone (bidirectional, non-causal) and the DiT (diffusion).
- With `--model-type=transformer`, backbone cos_sim degraded to ~0.547 due to
  NxDI's flash-attention kernel being activated (~3% error per layer × 16 layers).
- `-O1` uses the standard XLA matmul+softmax path, giving exact bfloat16 results.

**ViT** uses `--auto-cast=matmult --optlevel 3 --model-type=unet-inference`:
- `unet-inference` is the correct model type for vision encoders.

## TP=8 Design Rationale

- trn1.32xlarge has 32 NeuronCores, organized as 4 groups of 8.
- TP=8 is the standard NxDI TP degree for models with divisible head counts.
- Backbone: 16 Q-heads / 8 = 2 Q-heads per rank; 8 KV-heads / 8 = 1 KV-head per rank.
- VL self-attn: 32 heads / 8 = 4 heads per rank.
- DiT: 32 heads / 8 = 4 heads per rank.
- ColumnParallelLinear splits output dimension; RowParallelLinear all-reduces.
- Weight sharding verified: Q-proj [256, 2048] = 2048/8 per shard. ✓

## Known Deviations

1. **DiT cross-attention masking**: All even-indexed blocks attend to ALL 256
   conditioning tokens (Neuron), vs. alternating image/text masks (HF AlternateVLDiT).
   The image_mask is data-dependent and can't be a static compiled buffer.
   Impact: cos_sim=0.999930 — negligible.

2. **ViT static attention patch**: The Qwen3-VL ViT uses `cu_seqlens`-based packed
   attention with `torch.split(..., lengths.tolist())` — data-dependent and
   incompatible with neuronx-cc. Patched to standard full-sequence attention.
   For a single image (256 patches), this is numerically identical.
   ViT cos_sim=0.998119 (CPU vs NEFF).

3. **ViT deepstack outputs dropped in NEFF**: The deepstack intermediate features
   (at layers 5, 11, 17) are not returned by the compiled ViT NEFF. They are
   unused in the GR00T inference pipeline (backbone uses only the final merger output).

## ISA Kernel Settings (trn1)

`attn_kernel_enabled=False` is correctly set in `_make_backbone_inference_config()`.

NeuronConfig defaults on trn1:
- `attn_kernel_enabled=None` (default — resolves to False on trn1)
- `qkv_kernel_enabled=False`
- `qkv_nki_kernel_enabled=False`
- `mlp_kernel_enabled=False`
- `context_encoding_buckets=None`

The NKI flash-attention kernel (attn_kernel_enabled=True) is only beneficial on trn2+.
On trn1, it produces ~3% numerical error per layer (tiled online softmax vs exact).
Explicit `attn_kernel_enabled=False` ensures the standard XLA attention path is used.

## Directory Structure

```
groot-n1-port/
├── config_constants.py          # Architecture constants (verified vs checkpoint)
├── vlm_backbone_block.py        # 16-layer Qwen3-VL backbone (NxDI, TP=8)
├── vl_self_attention_block.py   # 4-layer VL self-attention (custom, TP=8)
├── dit_block.py                 # 32-layer DiT action head (custom, TP=8)
├── compile_all.py               # Compile backbone + vl_self_attn + dit
├── compile_vit.py               # Compile ViT (TP=1, patched attention)
├── run_inference.py             # Inference pipeline (load + run)
├── benchmark.py                 # End-to-end latency benchmark
├── benchmark_subgraph.py        # Per-subgraph latency breakdown
├── validate_neffs.py            # Correctness validation vs HF
├── open_loop_eval.py            # Open-loop MSE comparison
├── skills/scripts/
│   └── neuron_action_head_base.py   # Base classes for NxDI compilation
└── compiled/
    ├── backbone/                    # NEFF + 8x weight shards
    ├── vl_self_attn/                # NEFF + 8x weight shards
    ├── dit/                         # NEFF + 8x weight shards
    └── vit/                         # NEFF (TP=1, single file)
```
