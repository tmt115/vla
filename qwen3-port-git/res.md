# Port Reflection: Qwen3-VL → AWS Trainium (NxDI)

## What Was Done and How

The port followed the eight-deliverable structure prescribed by the skill. The sequence was:

- **D1–D2 (Vision)** — build `NeuronQwen3VLVisionAttention`, `NeuronQwen3VLVisionMLP`, `NeuronQwen3VLVisionBlock`, `NeuronQwen3VLVisionModel` by mapping from the HF `modeling_qwen3_vl.py` source. 14 vision block unit tests, 5 E2E CPU parity tests. The vision side was relatively clean because the ViT architecture (ViT blocks → PatchMerger → pos_embed) has a clear boundary and the NxDI attention base (`NeuronAttentionBase`) handled the rotary embedding abstraction.

- **D4–D5 (Text)** — build `NeuronQwen3VLAttention`, `NeuronQwen3VLDecoderLayer`, `NeuronQwen3VLTextModel`, `NeuronQwen3VLTextForCausalLM`. 14 text block unit tests, 9 E2E CPU parity tests. Heavier workaround load here: distributed init, `cpu_mode()` patching, and `NeuronBaseModel.forward()` bypass all had to be solved before a single test passed.

- **D3, D6** — `compile_vision.py` and `compile_text.py`, with the NKI CTE kernel bug discovery and fix.

- **D7, D8** — `smoke_test.py` with Tier 1/Tier 2 split, `notes.md`, `benchmark.py`, and `make_fake_weights.py` (unplanned but necessary because the model has no public checkpoint).

## Statistics

| Metric | Value |
|---|---|
| CPU tests passing | 43 / 43 |
| HF parameter count | 662 |
| Fake checkpoint params (after qkv split) | 770 |
| Fake checkpoint size | 16.4 GB bfloat16 |
| D3 vision NEFF | 94.3 MB |
| D6 text model NEFF | 29.6 MB |
| D6 vision model NEFF | 94.2 MB |
| Deviations from vanilla HF Qwen3VL | 8 (documented in notes.md) |

### Benchmark Results — trn1.32xlarge, tp=8, bf16, 784-patch input

| Component | P50 | P99 |
|---|---|---|
| D3 standalone vision encode | 97.0 ms | 100.6 ms |
| D6 vision encode (in pipeline) | 27.0 ms | 28.2 ms |
| D6 context encoding (CTE) | 454.7 ms | — |
| D6 full prefill (VE + CTE) | 481.8 ms | 483.0 ms |
| D6 single TKG step | 10.3 ms | 10.7 ms |
| Estimated 20-step decode | 687 ms | — |

## Issues Encountered

### 1. NKI CTE Attention Kernel Bug (NeuronX 2.24)

`attn_kernel_enabled=None` auto-enables the NKI flash-attention kernel (`attention_cte.py`) when `q_len ≥ 4096`. With `TEXT_SEQ_LEN=8192` this fires on every context-encoding pass and triggers a compiler internal error:

```
[NCC_INLA001] checkDMATranspose: DMACopy: transpose only supported for HBM->SB
Source Kernel "nkilib.core.attention.attention_cte.attention_cte", line 2258
```

Not documented anywhere. Diagnosed by reading the compiler traceback, then reading `NeuronAttentionBase` to find the `attn_kernel_enabled` flag. Fix: `attn_kernel_enabled=False`. This is the single most impactful silent trap for large-sequence models on this NeuronX version.

### 2. Qwen3-VL Has No Public Checkpoint

The model doesn't exist on HuggingFace yet. This wasn't known until three separate download attempts failed (hub returned 404). This invalidated the original assumption that benchmarking would use a real checkpoint. Required writing `make_fake_weights.py` from scratch — which then uncovered three separate key-naming bugs (below).

### 3. State Dict Key Transformation Chain (Three Separate Bugs)

This was the most time-consuming issue. The chain is:

```
HF safetensors
→ get_state_dict() strips _STATE_DICT_MODEL_PREFIX ("model.")
→ convert_hf_to_neuron_state_dict()
→ preshard_hook() renames HF-prefix keys to NxDI-prefix keys
```

Three bugs in this chain:

**Bug A — Vision qkv mismatch:** HF stores vision attention as fused `attn.qkv.weight [3456, 1152]`. The converter maps this to `attn.qkv_proj.Wqkv.weight` (still fused). But `fused_qkv` defaults to `False` in `NeuronConfig`, so `preshard_hook` expects split `q_proj`/`k_proj`/`v_proj`. The converter creates a key that `preshard_hook` never looks for. Fix: generate split HF-style keys in the fake checkpoint so the converter passes them through as `blocks.X.attn.q_proj.weight`, which `preshard_hook` then renames correctly.

**Bug B — Double `model.` prefix in text keys:** `Qwen3VLForConditionalGeneration` stores text as `model.language_model.model.layers.X....` After `get_state_dict` strips one `model.`, the text converter strips `language_model.`, leaving `model.layers.X....` But `preshard_hook` traverses from `NeuronQwen3VLTextModel` directly (no `model.` wrapper), so it expects `layers.X....` Fix: remap `model.language_model.model.` → `model.language_model.` in the fake checkpoint.

**Bug C — `output_attentions` missing:** `_setup_func_config` in `model_base.py` reads `self.text_config.output_attentions`. HF's `PretrainedConfig` sets this in `__init__` but it's not always reflected into `vars()` depending on Python descriptor mechanics. Fix: explicitly set after config construction.

### 4. `torch.jit.load` Requires Neuron C++ Runtime

The smoke test originally used `torch.jit.load` to verify NEFF integrity. This throws:

```
Unknown type name '__torch__.torch.classes.neuron.LayoutTransformation'
```

unless the Neuron C++ extensions are registered, which only happens after `app.load()`. File size ≥ 1 MB is a sufficient proxy for a valid compiled artifact.

### 5. CPU Test Infrastructure

Three separate issues hit in sequence during text-model testing:

- `cpu_mode()` returns `False` on trn1 hardware → wrong RMSNorm class → XLA tensor assertion
- `NeuronBaseModel.__init__` calls `SPMDRank` → requires `torch.distributed` even for rank-0-only tests
- `NeuronBaseModel.forward()` is the full inference-serving API; unit tests must bypass it with a manual sub-module chain

None of these are documented. Each required reading NxDI internals to diagnose.

## What Was Inefficient

The key transformation debugging loop was the biggest time sink. Three separate bugs were in the same pipeline (HF keys → NxDI keys), but each only manifested after the previous was fixed. A utility that printed the state dict at each transformation stage (raw HF, after `get_state_dict`, after converter, after preshard) would have collapsed this from ~hours to ~minutes.

The fake checkpoint required iteration. The meta-device approach to get shapes was correct, but the bugs in key naming weren't visible until actually running weight loading. Ideally, a validation pass that checks all NxDI model parameter names against the converted state dict keys before writing to disk would catch these immediately.

Smoke test was restructured twice. First version tried `torch.jit.load`; second version tried Tier 2 without a checkpoint. The Tier 1/Tier 2 split with graceful fallback should have been the first design.

The `bench_vision` Phase A latency (97 ms) vs D6 Phase A (27 ms) discrepancy was never fully explained. The D3 NEFF outputs shape `(1, 8192, 3584)` — batch × TEXT_SEQ_LEN × TEXT_HIDDEN_SIZE — which suggests D3 is doing more than pure vision encoding (possibly scattering features into a full text-length buffer). This should have been investigated more carefully rather than accepted.

## Why CTE Latency Is High (454.7 ms)

This deserves careful analysis. 454 ms for a single prefill pass is slow for a 7B model.

**Flash attention is disabled.** `attn_kernel_enabled=False` forces the standard XLA attention path, which materializes the full `(seq_len × seq_len)` attention matrix in HBM. At `seq_len=8192` and 28 layers, that's `28 × 8192² × 2 bytes = ~3.8 GB` of attention maps being written and read. The NKI `attention_cte` kernel (flash attention equivalent) uses tiled HBM→SRAM computation and avoids materializing the full matrix. Once NeuronX 2.25+ fixes the DMACopy transpose bug, re-enabling this will likely cut CTE time by 40–60%.

**Single text bucket at `seq_len=8192`.** Every prefill pads to 8192 tokens regardless of actual input length. A 32-token prompt runs the same 8192-token graph as a 8192-token prompt. A multi-bucket schedule — e.g. `[512, 1024, 2048, 4096, 8192]` — would reduce average CTE latency dramatically for typical inputs.

**tp=8 AllReduce overhead.** After each attention output projection and each MLP, a ring AllReduce runs across 8 NeuronCores. At `seq_len=8192`, the AllReduce payload is `8192 × 3584 × 2 bytes ≈ 56 MB` per layer × 28 layers. The ring latency on trn1 intra-node interconnect is low but not zero, and this is a synchronous blocking call per layer.

**No KV cache during prefill.** CTE writes the KV cache for all 8192 positions. That's `28 layers × 2 (K+V) × 4 heads (GQA, num_kv_heads=4) × 8192 × 128 × 2 bytes ≈ 1.5 GB` written per prefill pass.

The 27 ms vs 97 ms D3/D6 vision discrepancy also suggests the D3 NEFF is doing unnecessary work (possibly scattering into a full text-length output tensor), which inflates the standalone vision number but doesn't affect the production D6 pipeline.

## Ideas for Improving This Port

1. **Add text sequence length bucketing.** `buckets=[512, 1024, 2048, 4096, 8192]` instead of `[8192]`. This is the single highest-ROI change for real workloads.

2. **Re-enable NKI flash attention once NeuronX 2.25+ is available.** Remove `attn_kernel_enabled=False` and verify the compiler bug is fixed. Benchmark both with and without.

3. **Investigate D3 output shape.** The `(1, 8192, 3584)` output from the standalone vision NEFF suggests it's padding to `TEXT_SEQ_LEN`. If D3 can be made to output `(196, 3584)` directly, the standalone vision encode time drops and the shape-validation in `smoke_test.py` becomes meaningful.

4. **Add a key-mapping validation utility.** A small script that loads the fake checkpoint, runs it through `convert_hf_to_neuron_state_dict`, then compares the resulting keys against `model.named_parameters()` would catch all three key-naming bugs instantly without a full `app.load()` run.

5. **Extend benchmark to all vision buckets.** Currently only `bucket=784`. The 3136 and 12544 buckets give the scaling curve and reveal whether CTE time grows linearly with vision tokens.

6. **Benchmark with real vs fake weights.** For vision: rotary embeddings and attention softmax distributions differ significantly with random weights. The fake-weight benchmark is accurate for graph structure but may miss bandwidth effects from non-uniform attention patterns.

## Suggestions for the Skill (General PyTorch → NxDI)

These are architecture-agnostic lessons from this port that should feed back into the skill template:

1. **Add a state dict key transformation debugging section.** The chain `HF → get_state_dict prefix strip → convert_hf_to_neuron_state_dict → preshard_hook` is where most bugs live. The skill should include a debugging snippet:

   ```python
   sd = cls.get_state_dict(ckpt_path, cfg)
   print(sorted(sd.keys())[:20])  # inspect intermediate keys
   ```

   and explicitly warn: "check for leftover `model.` prefixes, fused vs split qkv, and any nested module path differences before running `load()`."

2. **Document `fused_qkv=False` default.** `NeuronConfig` defaults `fused_qkv=False` (split q/k/v). Many HF models store fused QKV. The skill should explicitly flag this: if the HF model has fused QKV, the converter must either split it or `fused_qkv=True` must be set. Otherwise `preshard_hook` silently fails to find the expected keys.

3. **Add an `attn_kernel_enabled` checklist item.** Any model with `seq_len ≥ 4096` should explicitly set `attn_kernel_enabled=False` and note the NeuronX version when this can be safely removed. This is not in any NxDI documentation.

4. **Two-layer fake checkpoint pattern should be in the template.** The skill currently treats fake checkpoints as trivial. This port showed there are two distinct fake checkpoint needs: (a) a minimal stub containing only the weights loaded at `__init__` time (for compilation — often just a positional embedding), and (b) a full random-weight checkpoint matching all NxDI-transformed key names (for benchmarking). The skill should codify both, with the validation step.

5. **Smoke test template: Tier 1 must not require hardware.** File-size checks (≥ 1 MB) are sufficient for NEFF integrity since `torch.jit.load` on Neuron `.pt` files requires the C++ runtime. The skill's smoke test template should use this pattern instead of `jit.load`, and clearly separate artifact-only checks (Tier 1) from inference checks (Tier 2).

6. **CPU test workarounds section.** The distributed-init and `cpu_mode()` patches are not model-specific — they apply to any NxDI text model with `NeuronBaseModel`. The skill should include a standard boilerplate fixture block for CPU tests that handles both.

7. **`output_attentions` / `output_hidden_states` guard.** `_setup_func_config` reads these from `text_config` but they are not guaranteed to exist if the HF config dict doesn't populate them. Add this as a standard line in the config-building function:

   ```python
   for attr in ("output_attentions", "output_hidden_states"):
       if not hasattr(cfg.text_config, attr):
           setattr(cfg.text_config, attr, False)
   ```

## NxDI Adherence

No meaningful deviations from NxDI. Every component (`NeuronAttentionBase`, `GQAQKVColumnParallelLinear`, `ColumnParallelLinear`, `RowParallelLinear`, `NeuronApplicationBase`, `NeuronBaseForCausalLM`, `NeuronBaseModel`) was used as intended. `NeuronLlamaMLP` was reused for the text decoder MLP — legitimate reuse since Qwen3's MLP is identical to LLaMA's SwiGLU. The three-dimensional MRoPE position IDs are the only significant non-standard interface, and those come from the HF model contract, not from departing from NxDI patterns.

The `attn_kernel_enabled=False` workaround is a bug avoidance, not a deviation — the kernel is NxDI-native, just broken on NeuronX 2.24 for this sequence length.
