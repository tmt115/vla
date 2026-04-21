# SmolVLA → Trainium Port: Retrospective

## What Was Ported

SmolVLA is a vision-language-action model composed of three distinct subgraphs:

| Subgraph | Params | Architecture |
|---|---|---|
| SigLIP vision encoder | ~86M | 12-layer ViT + pixel shuffle connector |
| SmolLM2 VLM backbone | ~500M | 16-layer GQA transformer (15/5 heads, hidden=960) |
| Action expert (denoiser) | ~100M | 16-layer interleaved self+cross-attn, hidden=720 |

Each subgraph was reimplemented from scratch in PyTorch, weight-mapped from the HuggingFace checkpoint, and compiled to a NEFF via `torch_neuronx.trace()` wrapped inside `ModelWrapper` subclasses.

---

## Port Phases

| Phase | Work |
|---|---|
| 1–2 | Architecture reimplementation of all three blocks |
| 3 | Unit tests (shape, dtype, output matching against HF reference) |
| 4 | Weight mapping from HF checkpoint to Neuron state dicts |
| 5 | NEFF compilation + benchmark |

Phase 5 alone spanned multiple sessions and required fixing five separate bugs before all three NEFFs compiled cleanly and performed well.

---

## Final Benchmark Results

Tested on a single Trainium instance, batch_size=1, 10 denoising steps, 20 warmup + 50 timed runs:

| Subgraph | Mean | Median | p95 | Std |
|---|---|---|---|---|
| Vision encoder | 9.5 ms | 9.5 ms | 9.5 ms | 0.0 ms |
| VLM backbone | 2.7 ms | 2.6 ms | 2.7 ms | 0.0 ms |
| Expert step (×1) | 2.7 ms | 2.7 ms | 2.8 ms | 0.0 ms |
| Expert loop (×10) | 27.7 ms | 27.7 ms | 27.9 ms | 0.1 ms |
| **End-to-end** | **40.1 ms** | **40.0 ms** | **40.2 ms** | **0.5 ms** |

**Throughput: 24.9 inferences/sec**

Variance is essentially zero — the Neuron runtime is fully deterministic once warmed up.

---

## Compiled NEFF Sizes

| Subgraph | Size |
|---|---|
| vision_encoder.pt | 160 MB |
| vlm_backbone.pt | 242 MB |
| action_expert.pt | 368 MB |

---

## Bugs Encountered

### 1. `neuronx-cc` errno 36 — Filename Too Long (VLM backbone)

**Symptom:** `RuntimeError: neuronx-cc failed with 70` and `[Errno 36] File name too long: 'value_sg0000_constant.144-10768-13172-...(68 node IDs)..._CRSM.npy'`

**Root cause:** `apply_rope()` called `torch.arange(d_half)` and `max_wavelength ** freq_exponents` dynamically inside `forward()`. With 16 transformer layers, the compiler encountered 16 identical constant tensors and merged them into a single constant node whose debug filename included all 68 parent node IDs — exceeding the 255-char filesystem limit.

**Initial wrong fix:** Changed the 16-input `torch.cat` in KV packing to sequential 2-input cats. This did not help — the real culprit was in RoPE, not KV packing.

**Actual fix:** Replaced `apply_rope(x, position_ids)` with `apply_rope_tables(x, sin_table, cos_table)` and pre-computed sin/cos buffers in `__init__` via `register_buffer()`. One static computation at init time, zero dynamic tensor creation in `forward()`.

**Lesson:** Any `torch.arange`, `torch.linspace`, or math over `torch.arange` inside `forward()` that runs once per layer will hit this if the model has enough layers. Pre-compute as registered buffers.

---

### 2. `strict=True` Failure for Backbone and Expert

**Symptom:** `Missing key(s) in state_dict: 'backbone.rope_sin', 'backbone.rope_cos'` (and later `rope_sin`, `rope_cos`, `self_attn_mask` for the action expert)

**Cause:** After adding the RoPE buffer fix and the static self-attn mask buffer, the computed buffers were not in the HF checkpoint, so strict weight loading rejected them.

**Fix:** Changed to `strict=False` with an assertion that only buffer keys (`rope_*`, `self_attn_mask`) were missing.

---

### 3. `load_state_dict` Missing `**kwargs`

**Symptom:** `TypeError: NeuronSmolVLADenoisingWrapper.load_state_dict() got an unexpected keyword argument 'assign'`

**Cause:** `torch_neuronx` internally calls `load_state_dict(..., assign=True)` during tracing. The custom override only accepted `(self, state_dict, strict=True)` and forwarded nothing else.

**Fix:** Added `**kwargs` to the override and passed them through to `nn.Module.load_state_dict`.

---

### 4. Silent Exception Swallowing in `compile_denoiser`

**Symptom:** `compile_denoiser()` returned without error, but no `.pt` file was written. Bugs 2 and 3 above were both hidden by this.

**Cause:** The method had `except Exception: pass` around the compile call.

**Fix:** Removed the try/except entirely. A compilation failure should always propagate.

---

### 5. Dynamic Constants in Action Expert Forward (performance bug)

**Symptom:** Action expert loop 28.7ms vs ~20ms on an equivalent port using direct `torch_neuronx.trace()` on the same SmolVLA architecture.

**Root cause:** Same dynamic constant pattern as bug 1, but in the action expert:
- `torch.arange(ACTION_CHUNK_SIZE)` — position IDs computed per step
- `torch.ones(B, 1, 50, 50)` — self-attention mask computed per step
- `apply_rope()` — `torch.arange(d_half)` + `timescale` computed per step

With 10 denoising steps, all three ran 10 times, creating duplicate constant nodes. This didn't crash (fewer repetitions than the 16-layer VLM case) but bloated the NEFF's constant handling and added overhead.

**Fix:** Pre-computed `rope_sin`, `rope_cos`, and `self_attn_mask` as registered buffers in `__init__`. Result: 28.7ms → 27.7ms per 10-step loop.

---

### 6. Background Process Silent Death

A compile job ran for 9.5 hours before I noticed the process was dead. The task notification system reported the job as still running even though no Python or neuronx-cc process existed. This wasted significant wall-clock time.

---

## Inefficiencies in This Port

- **Too many compilation attempts:** Five separate background jobs before clean benchmarks. Each attempt is 1–60 min. Better pre-compile validation would have caught all three errors immediately.
- **Sequential debugging:** Each bug only became visible after a full compile attempt. A dry-run mode that validates the model on CPU (without calling neuronx-cc) would have surfaced errors much faster.
- **Redundant recompile of vision:** `compile_all.py` does not skip already-compiled NEFFs.
- **The sequential-cat fix was a distraction:** Time spent on a fix that didn't address the root cause. The errno 36 error message points to a constant node — investigation should have started with dynamic constant creation, not data flow.
- **Dynamic constants in action expert were missed during initial implementation:** The same pattern that caused the VLM errno 36 crash was present in the action expert's forward pass but didn't crash (fewer layers = fewer duplicates), so it went unnoticed until benchmarking against a reference.

---

## Why End-to-End Latency Is Higher Than the Reference Port (~20ms)

Another SmolVLA port using direct `torch_neuronx.trace()` on the action head (same architecture, same hardware) achieved ~20ms. This port achieves 27.7ms for the expert loop alone, 40.1ms end-to-end.

**The dynamic constants fix (bug 5) recovered 1ms (28.7ms → 27.7ms) but did not close the gap.**

The remaining ~8ms difference on the expert loop is likely explained by:

1. **Conditioning format.** This port packs VLM KV as `[B, 113, 5120]` and reshapes it inside `forward()` on every step. The reference port may have passed conditioning as pre-split K/V tensors, allowing the compiler to better fuse the cross-attention projections.

2. **Compiler flags.** This port uses `--enable-mixed-precision-accumulation` for the expert, which forces float32 accumulators on matrix multiply. The reference port may not have used this flag. Removing it would be worth testing.

3. **The vision encoder adds 9.5ms per inference.** If the reference port measured only the action head, the 20ms comparison is apples-to-oranges. If it included vision+VLM, then 20ms total implies ~7.8ms for the expert loop (10 steps = 0.78ms/step), which would be unusually fast for 16 attention layers.

**Bottom line:** This port is architecturally correct and fully functional. The remaining latency gap is worth investigating via flag tuning and conditioning format changes, but is not a blocker.

---

## NxDI Deviations and Justification

The port uses NxDI infrastructure at the layer level but deviates at the application level:

**Used:**
- `ColumnParallelLinear` / `RowParallelLinear` for all linear layers, with a `parallel_state` branch that falls back to `nn.Linear` on CPU and when TP is not initialized (which is the case during `torch_neuronx.trace()` — so all compiled NEFFs use plain `nn.Linear` ops)
- `ModelWrapper` subclasses to own `torch_neuronx.trace()` calls
- `NeuronDenoisingWrapper` and `NeuronActionHeadBase` from the skill's base classes — subclassed for interface compatibility (`ConditioningContract`, `generate_actions()`, `get_conditioning_contract()`)

**Bypassed:**
- `ModelWrapper.__init__()` — requires an `InferenceConfig` with LLM-specific attributes (KV cache size, etc.) that don't apply to these subgraphs. Bypassed with `nn.Module.__init__(self)`.
- `NeuronApplicationBase.__init__()` — same reason.
- `NeuronAttentionBase` — the GQA attention subclassing interface is designed around autoregressive decoding with KV cache management. The VLM backbone does a full prefix pass (no cache), and the action expert uses a fixed-length denoising input. Custom attention implementations were simpler and more appropriate.

**Justification:** NxDI's high-level abstractions are designed around autoregressive LLM decoding. This port does prefix-only and denoising inference. Using the lower-level primitives (parallel linear layers, ModelWrapper) gives Neuron-compatible compilation while avoiding fighting the framework. The `NeuronDenoisingWrapper`/`NeuronActionHeadBase` subclassing was retained for interface compatibility with the skill's scaffolding, even though the `__init__` logic was bypassed.

---

## Ideas for Improving This Port

**1. Test without `--enable-mixed-precision-accumulation` on the expert.**
This flag forces float32 accumulators and likely accounts for several ms of the latency gap. It trades accuracy for speed — worth benchmarking without it to quantify the cost.

**2. Amortize the VLM/vision pass.**
The vision + VLM backbone (12.2ms combined) only needs to rerun when the image or language instruction changes. Caching conditioning across denoising calls would drop end-to-end to ~28ms.

**3. Try a different conditioning format.**
Pass K and V as separate pre-split tensors rather than packed `[B, 113, 5120]`. This avoids the reshape inside forward and may allow the compiler to better fuse cross-attention projections.

**4. Enable TP>1 for the VLM backbone.**
At TP=8 the VLM backbone would split across NeuronCores. Current VLM latency is 2.7ms so this is low priority, but would matter at larger batch sizes.

**5. Add a skip-if-compiled check to `compile_all.py`.**
Check `os.path.exists(save_path)` before each subgraph and skip with a message. Avoids recompiling unchanged subgraphs.

**6. Add pre-compile CPU validation.**
Run a CPU forward pass with the same example inputs before calling neuronx-cc. Catches dtype, shape, and kwargs errors immediately instead of after a multi-minute compile.

---

## Suggestions for the Port Skill / MDs (General)

These apply to any PyTorch → Trainium port, not SmolVLA specifically.

**1. Warn about dynamic constant creation in `forward()`.**
`torch.arange`, `torch.linspace`, `torch.ones`, `torch.zeros`, and math over these inside `forward()` will cause errno 36 if the model has many layers, and will silently degrade performance even if they don't crash. The skill/MD should include a pre-compile checklist: *"Does your forward() create any range, frequency, or mask tensors? Move them to `__init__` as `register_buffer`."*

**2. Document the `**kwargs` requirement for `load_state_dict` overrides.**
Any custom `load_state_dict` must accept and forward `**kwargs`. `torch_neuronx` passes `assign=True` internally; not forwarding it causes a `TypeError` that, if swallowed by a try/except, produces a silent missing-output-file failure.

**3. Prohibit bare `except Exception: pass` around compile calls.**
This is the single most expensive mistake in this port — it hid two separate bugs behind a missing file. The skill should explicitly state that compile calls must not be wrapped in exception-swallowing try/except.

**4. Add a "skip if already compiled" pattern to the compile script template.**
Default template should check `os.path.exists(save_path)` and skip with a message. Recompiling unchanged subgraphs wastes significant time.

**5. Note that `ModelWrapper.__init__` requires `InferenceConfig`.**
When bypassing it via `nn.Module.__init__(self)`, document why. The skill should note this is valid for non-autoregressive subgraphs.

**6. Add background process liveness check guidance.**
Background compile jobs can die silently. The skill should recommend checking `ps aux | grep neuronx` after a few minutes to confirm the process is alive, not assuming a long-running background job is still working.

**7. Note that `parallel_state` is not initialized during `torch_neuronx.trace()`.**
If a model uses `ColumnParallelLinear`/`RowParallelLinear` with a `parallel_state` branch, the trace will use the `nn.Linear` fallback — so the compiled NEFF is always single-device regardless of the branch. This means TP>1 requires a different compilation path (NxDI's distributed trace infrastructure), not just the parallel layer definitions.
