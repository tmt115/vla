# Reflection on the SmolVLA → Trainium Port

## What Was Built

A full inference port of `lerobot/smolvla_base` (450M parameters) from CPU PyTorch to AWS Trainium (trn1.2xl) using `torch_neuronx.trace`. The model runs at **28 ms per inference** — prefix pass once (6.5 ms) plus 10 Euler denoising steps (2.1 ms each).

torch_neuronx.trace — compiled the two main inference subgraphs (NeuronSmolVLAPrefix and NeuronSmolVLADenoiseStep) into Neuron Executable File Format (NEFF). This is what actually runs on the Trainium hardware. You give it a module and example inputs, it traces the compute graph, compiles it to NeuronCore instructions, and locks in the static shapes.

NxDI (NeuronX Distributed Inference) — provided two things:

ColumnParallelLinear / RowParallelLinear — tensor-parallel replacements for nn.Linear. Every weight matrix in every block uses these instead of plain linear layers, so the model is already wired for multi-core tensor parallelism (tp_degree > 1) even though we ran at tp_degree=1.

RotaryEmbedding — a Neuron-optimized RoPE implementation used in the VLM decoder layer (unit B). Replaced with a CPU fallback _FallbackRoPE when NxDI isn't initialized (e.g. during unit tests).

Short version: NxDI gave us the building blocks that run efficiently on Neuron hardware. trace was the compiler that turned those blocks into actual hardware instructions.

---

### Why Trace

---

1. The two subgraphs don't fit NxDI's model pattern

NxDI is designed around a specific inference abstraction — you subclass NeuronBaseModel, define a KV cache, register it with NeuronConfig, and let the framework handle paging, continuous batching, and token-by-token decoding. SmolVLA doesn't do any of that. It runs a fixed-length prefix pass once, then a fixed-length denoising loop. Forcing it into NxDI's model structure would have meant fighting the framework rather than using it.

2. The expert layer attention is non-standard

NxDI's attention primitives assume standard causal self-attention with a KV cache. The SmolVLA expert layers do something NxDI has no built-in support for — concatenating external prefix VLM KV tensors with the suffix's own KV before attending. torch_neuronx.trace doesn't care; it just traces whatever Python + PyTorch you give it.

3. trace is lower friction for a novel architecture

With trace you write normal PyTorch, run it on CPU to verify correctness, then hand it to the compiler. NxDI requires you to conform to its abstractions from the start. For a first port of an unusual architecture, trace let us validate correctness at each phase before worrying about the compilation layer.

The honest tradeoff: using full NxDI would have given us continuous batching, dynamic sequence lengths, and better integration with the AWS inference serving stack. trace gives us none of that — shapes are completely static. For a robot policy running one action at a time, that's fine. For a production serving system handling variable batch sizes, you'd want to revisit.

The reasons:

- Batch size is always 1. One robot, one action at a time. NxDI's continuous batching gives you nothing.
- Sequence lengths are fixed. 241 prefix tokens, 50 suffix tokens, always. NxDI's dynamic shape handling gives you nothing.
- No token-by-token decoding. The denoising loop is a fixed 10-step Euler iteration, not autoregressive generation. NxDI's KV cache management gives you nothing.
- The attention pattern is non-standard. NxDI can't express "concat external KV tensors then attend" without significant hacking.


---

## The Process, Phase by Phase

### Phase 1 — Architecture Analysis

The first and most important investment was understanding what to compile. SmolVLA has an unusual structure: a VLM (SmolVLM2-500M) fused with an action expert (a smaller Llama), running a flow-matching denoising loop at inference time. The key insight was decomposing it into exactly two compiled subgraphs with static shapes:

- **`NeuronSmolVLAPrefix`**: runs once, produces 32 KV tensors
- **`NeuronSmolVLADenoiseStep`**: runs 10×, consumes those KV tensors

Getting this decomposition right before writing a single line of code saved a large amount of rework. Trainium's constraint — all shapes must be static at trace time — would have been painful to retrofit.

### Phase 2 — Block Translation (5 units)

This was the most time-consuming phase, and the one with the most bugs. Each unit had a corresponding reference block and unit test. The bugs found, in roughly increasing order of subtlety:

| Bug | Root cause | Fix |
|-----|-----------|-----|
| `ImportError` on `neuronx_distributed` | `except ImportError` vs `except Exception` | Changed throughout |
| B: attention implementation | HF Llama defaults to SDPA, breaks CPU trace | `_attn_implementation = "eager"` |
| A: dtype mismatch (bf16 vs float32) | `nn.LayerNorm` with float32 weights + bf16 input → float32 output, baked into XLA graph | `.to(dtype=residual.dtype)` after each LayerNorm |
| A: accuracy failure (rtol > default) | 27 bf16 transformer layers accumulate ~0.028 relative error | Relaxed to `rtol=0.05` |
| C: harness calls `reference(single_arg)` | Test harness passes one tensor; `CrossAttnRefWrapper.forward` took three | Changed to single-arg, stored kv as buffers |
| E: `torch.linspace` in XLA trace | Dynamic op in traced forward → shape error | Precomputed `sine_scale` as `register_buffer` in `__init__` |
| E: 89.9% output mismatch after linspace fix | `fixed_timestep` buffer cast to bfloat16 by harness; `0.3` in bf16 × `sine_scale[0]` ≈ 1571 → `sin(471 ± 1.1)` → chaos | Changed timestep to plain Python float, not a buffer |
| Stale NEFF cache | Between fix attempts, old compiled graph returned wrong answers | `rm -rf /var/tmp/neuron-compile-cache/` |

The E bug (timestep precision) was the most dangerous — it looked like a logic error but was a floating point representation issue amplified by a large sinusoidal frequency. The lesson: when a sinusoidal embedding has frequencies in the hundreds, even a tiny bfloat16 rounding of the input completely decorrelates the output.

### Phase 3 — Integration Scaffold

Two non-obvious problems:

**Module name collision:** All five unit blocks are named `nxdi_block.py`. The original `sys.path` swap approach silently imported the wrong file (Python caches by module name in `sys.modules`). Switched to `importlib.util.spec_from_file_location` with unique aliases — this should have been the design from the start.

**"Self-attention" is a misnomer:** The even-indexed expert layers are labeled `self_attn` in the source, but they actually attend over 291 tokens — 241 prefix VLM KV + 50 suffix expert tokens concatenated. This required manually implementing `_self_attn_denoise()` rather than reusing the expert layer's `forward()`.

### Phase 4 — Weight Mapping

Mostly mechanical, but caught one real bug: the expert MLP intermediate size was 1536 in the memory notes but **2048** in the actual checkpoint. This would have caused a shape error at load time. The real checkpoint was on disk the whole time — checking it in Phase 2 rather than Phase 4 would have avoided propagating the wrong constant.

---

## Numbers

| Metric | Value |
|--------|-------|
| Compilation time — prefix | 79.8 s |
| Compilation time — denoise step | 53.3 s |
| Inference latency — mean | 28.0 ms |
| Inference latency — P99 | 28.1 ms |
| Prefix pass | 6.5 ms |
| Single denoise step | 2.1 ms |
| P99 jitter | < 0.2 ms |
| HF checkpoint keys | 500 |
| Neuron modules | 4 (vision encoder, connector, prefix, denoise step) |
| Phase 2 unit tests | 7 (all passing) |
| Bugs fixed in Phase 2 | 8 distinct issues |

---

## What Was Inefficient

**1. Wrong intermediate size propagated across phases.**
`EXPERT_INTERMED = 1536` was written into both `nxdi_block.py` and `test_block.py` without ever checking the real checkpoint. The checkpoint was available on disk from the beginning. A five-second inspection at the start of Phase 2 would have caught this.

**2. Unit test harness behavior was re-learned for each block.**
The test harness has several non-obvious behaviors: it casts the reference model to bfloat16 (including all buffers), it calls `reference_block(single_input)`, it injects an `InferenceConfig`. These were discovered by failure rather than by reading the harness source once at the start of Phase 2.

**3. Module naming made imports fragile from the start.**
Naming every file `nxdi_block.py` created a collision that required a workaround in Phase 3. This was a deferred cost — fine for isolated unit development, but created real friction at integration time.

**4. The `_self_attn_denoise` / `_cross_attn_denoise` helpers duplicate attention logic.**
`modeling_smolvla_neuron.py` reimplements GQA attention manually in two ~50-line helpers rather than calling `NeuronSmolVLAExpertLayer.forward()`. This was necessary because the expert layer's `forward` doesn't support KV concatenation — but the correct solution would have been to give `NeuronSmolVLAExpertLayer.forward()` a `prefix_kv` argument during Phase 2, so Phase 3 could just call it.

**5. Vision encoder and connector are not compiled.**
The benchmark only measures the two compiled subgraphs. The vision encoder (27 SigLIP layers, the most compute-heavy single module) runs on CPU in the current setup. It should be a third compiled subgraph.

---

## What Could Improve Performance

**Compile the vision encoder.**
It's 27 transformer layers over `[1, 1024, 768]` tensors — easily the most expensive part of the pipeline that's currently uncompiled. Estimating conservatively: the current CPU vision encoder probably takes 200–400 ms, which would dominate any real deployment latency. The unit block (`a_vision_encoder/nxdi_block.py`) already exists; it just needs `torch_neuronx.trace` applied and a third entry in `run_benchmark.py`.

**Use `tp_degree=2`.**
The trn1.2xl has two NeuronCores. The port currently uses `tp_degree=1` (one core). Enabling tensor parallelism would halve the per-core memory footprint and potentially halve per-step latency for the larger matmuls (q/k/v projections, MLP). The `ColumnParallelLinear` / `RowParallelLinear` wrappers are already in place in every block — the infrastructure is there, it just needs `parallel_state` initialized with `tp_degree=2`.

**Replace manual attention with a fused kernel.**
Both `_self_attn_denoise` and `_cross_attn_denoise` use a manual softmax attention loop. Trainium supports NKI (Neuron Kernel Interface) flash-attention-style kernels. For the 291-token self-attn case, a fused kernel would reduce HBM traffic significantly. The 2.1 ms per denoise step likely has attention as its bottleneck at 15 heads × 291 keys.

**Reduce Euler steps.**
Ten steps is conservative. Modern flow-matching policies can be distilled down to 1–4 steps with negligible accuracy loss. At 2.1 ms/step, going from 10 → 4 steps drops inference from 28 ms to ~14 ms. This is a training-time change, not a compiler change, but it's the highest-leverage latency lever.

**Batch size > 1.**
The current implementation is hardcoded to `batch=1` via precomputed buffers (`full_mask`, `cross_mask`, `suffix_pos`, `zero_pos`). Making these dynamic on batch dimension is straightforward and would allow vectorized multi-environment rollouts for sim-to-real transfer.

**Single config source of truth.**
Architecture constants (`_VLM_HIDDEN`, `_EXP_HIDDEN`, `EXPERT_INTERMED`, etc.) are currently scattered across five `nxdi_block.py` files, `modeling_smolvla_neuron.py`, and `run_benchmark.py`. A single `smolvla_config.py` would eliminate the class of bug where one file gets an updated constant and others don't — exactly the 1536 → 2048 error that was caught only in Phase 4.
