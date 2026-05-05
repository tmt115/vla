---
name: translate-model
description: Port models defined in PyTorch to run on AWS Trainium accelerators.
---

# AWS Trainium
AWS Neuron hardware consists of custom-designed machine learning accelerators optimized for deep learning workloads. 

At the heart of the Trn1 instance are 16 x Trainium chips (each Trainium include 2 x NeuronCore-v2). Trainium is the second generation purpose-built Machine Learning accelerator from AWS.

| **Category**        | **Specification**                                                                                                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Compute**         | Two NeuronCore-v2 delivering:<br>• 380 INT8 TOPS<br>• 190 FP16 / BF16 / cFP8 / TF32 TFLOPS<br>• 47.5 FP32 TFLOPS                                                                                          |
| **Device Memory**   | • 32 GiB device memory (model state storage)<br>• 820 GiB/sec memory bandwidth                                                                                                                            |
| **NeuronLink**      | NeuronLink-v2 chip-to-chip interconnect enabling:<br>• Efficient scale-out training<br>• Memory pooling across Trainium chips                                                                             |
| **Programmability** | • Dynamic shapes and control flow via NeuronCore-v2 ISA extensions<br>• User-programmable rounding mode (Round Nearest Even Stochastic Rounding)<br>• Custom operators via deeply embedded GPSIMD engines |


# NeuronX Distributed Inference

NxD Inference (where NxD stands for NeuronX Distributed) is an open-source PyTorch-based inference library that simplifies deep learning model deployment on AWS Inferentia and Trainium instances. It offers advanced inference capabilities, including features such as continuous batching and speculative decoding for high performance inference. Neuronx Distributed Inference includes a model hub and modules that users can reference to implement their own models on Neuron.

NxD Inference(NxDI) library offers the following benefits:
- Production ready models: NxD Inference provides production ready models like Llama-3.1, DBRX, and Mixtral that you can quickly deploy for high performance inference.
- LLM Inference Features: NxD Inference provides support for various LLM inference features like KV Cache, Multi-Head Attention (MHA), Grouped Query Attention (GQA), Flash Attention, Quantization, MoE , Continuous Batching and Speculative Decoding enabling high performance inference.
- Modular Design: Inference features in NxDI like KV Caching are implemented with a modular design, allowing developers to easily incorporate them into new models or customize and extend them.
- Distributed Strategies: NxD Inference enables distributing inference workload of large models across multiple NeuronCores in a single instance using Tensor parallelism and Sequence Parallelism. Pipeline parallelism and multi-node inference will be supported in future Neuron releases.
- Support for NKI Kernels: NxD Inference provides support for integrating custom NKI kernels on Trainium and Inferentia instances.

## Defining Models in NxDI
This guide demonstrates how to adapt an existing PyTorch model to run on Neuron with the NeuronX Distributed (NxD) Inference library. 

### 1. Define a NeuronConfig Class
Define a Neuron configuration class, which extends NeuronConfig. NeuronConfig includes Neuron-specific configuration parameters. In the config class for your model, you can define any additional Neuron-specific configuration parameters that your model requires.

```python
from neuronx_distributed_inference.models.config import NeuronConfig

class NeuronLlamaConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set any args/defaults
```
### 2. Define an InferenceConfig class

Define an inference configuration class, which extends InferenceConfig. InferenceConfig includes model parameters, such as those from a HuggingFace PretrainedConfig (like LlamaConfig). When users initialize your config, they can provide required attributes directly, or they can populate the config from a HuggingFace PretrainedConfig. You can also override get_required_attributes to enforce that certain attributes are present.

```python
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig

class LlamaInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronLlamaConfig
```

### 3. Define a Neuron model

This class is a subclass of NeuronBaseModel, which is a PyTorch module.

1. In this class, you provide implementations for setup_attr_for_model(self, config) and init_model(self, config).

    1. In setup_attr_for_model, set values for the following attributes. You can set these attributes from values in config and config.neuron_config.

        - self.on_device_sampling
        - self.tp_degree
        - self.hidden_size
        - self.num_attention_heads
        - self.num_key_value_heads
        - self.max_batch_size
        - self.buckets

    2. In init_model, initialize the modules that make up the model.
        -  For attention modules, extend NeuronAttentionBase, which provides a group query attention (GQA) implementation adapted to Neuron.
        - Replace linear layers (such as in attention and MLP) with Neuron parallel layers (RowParallelLinear and ColumnParallelLinear).
        - Replace embeddings with Neuron parallel embeddings (ParallelEmbedding)
        - Replace any other modules that require Neuron-specific implementations.

### 4. Define an application/task head

Define an application/task head. Applications includes causal LM, classification, and so on. This class extends a task-specific Neuron application head class (such as NeuronBaseForCausalLM), or the general NeuronApplicationHead class.

In this class, you provide an value for _model_cls which is the Neuron model class you defined.

You can also override any other functions as needed for your model, such as get_compiler_args(self) or convert_hf_to_neuron_state_dict(model_state_dict, neuron_config).

Note: This example demonstrates a simplified version of NeuronLlamaForCausalLM from the NxD Inference model hub.

```python
class NeuronLlamaForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronLlamaModel

    @classmethod
    def get_config_cls(cls):
        return LlamaInferenceConfig
```

# High Level Workflow

Translating a model will involve the following phases. Some phases are more involved and contain details in the linked documentation - load resources as needed during development.

**Note**: Activate the environment before running any code `source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`

**Routing:**
- Text-only LLM → proceed directly to Phase 1 below
- VLM (model accepts image inputs e.g. `pixel_values`, aspect ratios, vision masks) → read [reference/vlm_translation.md](reference/vlm_translation.md) first, then apply phases below
- VLA (model has an action head in addition to VLM inputs) → read [reference/vlm_translation.md](reference/vlm_translation.md) first, then read [reference/action_head_translation.md](reference/action_head_translation.md), then apply phases below

---

## Phase 0: Environment and Checkpoint Inspection

*This phase runs before any planning or translation. It is mandatory and must complete fully before Phase 1 begins.*

**Do not skip this phase.** Bugs introduced by unverified constants propagate through all subsequent phases and are expensive to fix late.

### Step 1 — Verify environment

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python -c "import neuronx_distributed_inference; print('NxDI OK')"
python -c "import torch_neuronx; print('torch_neuronx OK')"
```

Confirm `tests/block_testing_utils.py` is present:

```bash
ls tests/block_testing_utils.py
```

If missing, copy it from `scripts/block_testing_utils.py` before proceeding.

### Step 2 — Read the test harness

Read `tests/block_testing_utils.py` fully. Document the following in the plan:
- How `test_block_correctness` is called (signature, arguments)
- What dtype the harness casts models to (typically bfloat16, including all buffers)
- What calling convention it uses for the reference block (single-arg vs multi-arg)
- Whether it injects an `InferenceConfig` and how

**This information must be included in every Phase 2 subagent prompt.** Subagents that discover harness behavior through failure waste significant time.

### Step 3 — Inspect the real checkpoint

Load the model checkpoint and extract all architecture constants:

```python
import torch
from huggingface_hub import snapshot_download

# load config and state dict
state_dict = torch.load("path/to/model.bin", map_location="cpu")  # or .safetensors
for k, v in state_dict.items():
    print(k, v.shape)
```

Write every architecture constant to `config_constants.py` in the working directory. Name constants after the model's actual structure — for example:

```python
# config_constants.py — generated by Phase 0, do not edit manually
# All values read directly from checkpoint tensors, not from config.json alone.
HIDDEN_SIZE = ...
NUM_ATTENTION_HEADS = ...
NUM_KEY_VALUE_HEADS = ...
NUM_HIDDEN_LAYERS = ...
INTERMEDIATE_SIZE = ...       # always verify against checkpoint shape, not config.json
HEAD_DIM = ...
VOCAB_SIZE = ...
# Add any additional constants specific to this model's architecture
# e.g. NUM_EXPERTS, EXPERT_INTERMEDIATE_SIZE, VLM_HIDDEN_SIZE, ACTION_DIM, etc.
```

**All subsequent phases must import constants from `config_constants.py`. No hardcoded numbers anywhere else.**

### Step 4 — Write STATUS.md

Create `STATUS.md` in the working directory:

```markdown
# Port Status

## Phase 0 — Complete
- Environment verified: [date]
- Checkpoint inspected: [model path]
- config_constants.py written: yes
- Test harness calling convention: [document here]
- Test harness dtype casting: [document here]

## Phase 1 — Pending
## Phase 2 — Pending
## Phase 3 — Pending
## Phase 4 — Pending
## Phase 5 — Pending
## Phase 6 — Pending
```

Update `STATUS.md` at the end of every phase.

---

## Phase 1: Model Understanding and Planning

The orchestrator must **not** read source files or plan directly. It dispatches a single `plan` subagent that performs source exploration and returns the full Phase 2 execution plan.

### Step 1 — Dispatch a plan agent for exploration + work plan

Launch a `plan` subagent (thoroughness: "very thorough") with a prompt that instructs it to return one self-contained plan covering:

1. **Source model architecture inventory.** Read the model's PyTorch source and HuggingFace config. Identify every major block type present: attention (MHA/GQA/MQA), MLP, MoE routing and expert layers, embedding tables, normalization layers, positional encodings (RoPE, ALiBi, etc.), and any custom ops. Include file paths and class names for each block.

2. **Reference NxDI model.** Identify which reference model best matches the target architecture. **ALWAYS check the installed NxDI package first** before writing any custom code:

   ```bash
   ls /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/
   ```

   Known installed ports (check — versions may vary):
   - **Qwen3-VL text backbone**: `qwen3_vl/modeling_qwen3_vl_text.py` — `NeuronQwen3VLTextForCausalLM`. **If your backbone is Qwen3-VL: subclass this, override only `convert_hf_to_neuron_state_dict` for checkpoint prefix differences. Do NOT rewrite it.**
   - **Qwen3-VL vision encoder**: `qwen3_vl/modeling_qwen3_vl_vision.py`
   - **Text MoE**: `qwen3_moe/modeling_qwen3_moe.py`
   - **VLM (scatter integration)**: `pixtral/modeling_pixtral.py`
   - **VLM (cross-attention integration)**: `mllama/modeling_mllama.py`
   - **VLM (chunked vision flow)**: `llama4/modeling_llama4.py`
   - **Other (dense text)**: `llama/modeling_llama.py`

   If a matching port exists: subclass it, override `convert_hf_to_neuron_state_dict` for key differences, set `num_hidden_layers` to truncate if needed. Layer construction inside `NeuronBaseModel.init_model()` already happens after `parallel_state` is active — this is the correct pattern. Do NOT rewrite working NxDI code from scratch.

3. **NxDI compatibility assessment.** For each block, explicitly state whether it maps cleanly to an NxDI primitive. NxDI is designed for standard autoregressive LLM inference — models that deviate from this pattern will require partial or full `torch_neuronx.trace()`. Non-standard patterns to watch for include: denoising or diffusion loops, flow matching, non-autoregressive generation, fused multi-model architectures, custom KV cache layouts, dynamic control flow, and task heads that output non-token tensors (e.g. actions, embeddings, class logits). Document which subgraphs should use NxDI primitives and which should use `torch_neuronx.trace()`. Do not silently fall back — every `trace()` usage must be explicitly justified in the plan.

   **5M instruction limit.** If a single subgraph exceeds 5M compiler instructions (error `NCC_EVRF007`), split at block boundaries. Pattern: split at natural layer boundaries (e.g. after every N transformer blocks), pass intermediate tensors between parts, compile each part separately. Run `torch_neuronx.analyze()` before compilation to estimate instruction count.

   **ISA kernel availability by hardware.** ISA kernel settings differ by platform — verify before setting `attn_kernel_enabled`:

   | Platform | attn_kernel_enabled | qkv_kernel | mlp_kernel | Notes |
   |----------|--------------------|-----------:|------------|-------|
   | trn1 | False (explicit) | off | off | All ISA kernels off; enabling produces ~3% error per layer that compounds |
   | trn2 tp=2 | True (required) | required | off | QKV+Attn kernels required — ICE NCC_ITEN404 without them |
   | trn2 tp=4 | True (recommended) | recommended | off | MLP kernel hurts at small scale (-15.6%) |
   | inf2 | False (explicit) | off | off | Different compiler bug; all kernels off |

4. **Neuron substitution map.** For each block type found, map it to the corresponding NxDI primitive (`NeuronAttentionBase`, `RowParallelLinear`/`ColumnParallelLinear`, `ParallelEmbedding`, etc.) or mark as `trace()` with justification. Flag any blocks with no obvious NxDI equivalent.

5. **HuggingFace config attribute inventory.** List all fields from the model's `PretrainedConfig` that must be surfaced in `InferenceConfig.get_required_attributes()`. For each attribute, note whether it exists verbatim in `config.json` or is computed/renamed at Python object construction time (the latter must be handled in `add_derived_config`). **Cross-reference every constant against `config_constants.py` — do not use values from `config.json` alone.**

6. **Block partition.** Divide the model into independent translation units (one per Phase 2 subagent). Each unit must be self-contained (no shared mutable state). Typical partitions by model type:

   Standard LLM:
   - Attention subagent (attention block + KV cache integration)
   - MLP/FFN subagent (dense feed-forward layers)
   - MoE subagent (router + expert dispatch + expert layers), if applicable
   - Embedding & normalization subagent
   - Positional encoding subagent, if non-standard

   Vision/multimodal additions:
   - Vision encoder subagent (patch embedding + vision transformer layers)
   - Cross-modal connector subagent (projection, resampler, etc.)

   Action head (VLA only):
   - Action head subagent (denoising wrapper + cross-attention conditioning +
     timestep embedding) — follow reference/action_head_translation.md

   Non-standard inference (diffusion, flow matching, etc.):
   - Identify subgraph boundaries at natural static-shape boundaries
   - Each compiled subgraph must have fully static input/output shapes
   - Any loop or dynamic control flow runs in Python between compiled subgraphs

7. **Per-subagent instructions.** For each translation unit, specify:
   - The source PyTorch class(es) to translate (file path + class name)
   - The NxDI base class and primitives to use, OR explicit `trace()` justification
   - The output file name (named after the block, e.g. `attention_block.py` — never `nxdi_block.py`)
   - The test harness calling convention for this block (from Phase 0 Step 2)
   - Any flagged deviations or unsupported ops to watch for

8. **Integration contracts.** For each block, specify the exact input/output tensor shapes and dtypes it must satisfy so blocks compose correctly in Phase 3.

The orchestrator consumes this plan output directly to drive Phase 2.

### Step 2 — Update STATUS.md

```markdown
## Phase 1 — Complete
- Reference model: [path]
- NxDI compatibility: [summary of which blocks use NxDI vs trace()]
- Block partition: [list of translation units]
- Any non-standard patterns identified: [list]
```

---

## Phase 2: Block Translation and Unit Testing
*The orchestrator agent dispatches block-translator agents to translate all block partitions identified in Phase 1 in parallel. Each agent operates independently on its assigned block.*

Before launching subagents, confirm `tests/block_testing_utils.py` is present and `config_constants.py` exists.

Each subagent prompt must be constructed directly from the plan created in Phase 1. For each translation unit, the orchestrator extracts the relevant section of the plan and passes it as the subagent's full context. Each subagent receives: the source PyTorch implementation of its block, the integration contract, the test harness calling convention (from Phase 0), and a reference to the relevant NxDI primitives or `trace()` pattern.

**Important:** Make the execution workflow as clear as possible for each block-translator subagent. Each subagent should spend minimal time planning, spending most of its time implementing and debugging. The orchestrator must give it access to all necessary information needed to perform translation.

**Each subagent must:**

1. **Name the output file after the block.** Use descriptive names like `attention_block.py`, `mlp_block.py`, `vision_encoder_block.py`. Never use `nxdi_block.py` — this causes module name collisions at integration time.

2. **Import all constants from `config_constants.py`.** Never hardcode architecture constants (hidden sizes, number of heads, intermediate sizes, etc.). Every constant must come from `config_constants.py`.

3. **Implement the Neuron block class.** Subclass the appropriate NxDI base per the substitution map from Phase 1. Replace all standard PyTorch layers with Neuron parallel equivalents.

   **THE SINGLE MOST IMPORTANT RULE: Layer construction must happen in `load_module()`, NOT `__init__()`.**

   `ColumnParallelLinear` and `RowParallelLinear` check whether `parallel_state` is initialized at construction time. `parallel_state` is only initialized by `ModelBuilder` — which calls `load_module()` after setup. If you construct layers in `__init__()`, `parallel_state` is not yet active, every linear layer silently falls back to `nn.Linear`, and the NEFF runs as TP=1 regardless of what tp_degree is configured. The weights will not be sharded. Loading will fail with "Expected weight tensors for N ranks. Received 1."

   **CORRECT pattern — layers in `load_module()`:**
   ```python
   class MyNeuronWrapper(nn.Module):
       def __init__(self):
           nn.Module.__init__(self)
           self.model = None        # DO NOT construct here
           self._preload_sd = None

       def load_module(self):
           # ModelBuilder calls this AFTER parallel_state(tp_degree=N) is initialized
           # ColumnParallelLinear NOW uses real TP — parallel_state is active
           self.model = MyActualModel()
           if self._preload_sd is not None:
               self.model.load_state_dict(self._preload_sd, strict=False)
           self.model = self.model.bfloat16().eval()
   ```

   **WRONG pattern — produces silent TP=1, broken weight sharding:**
   ```python
   class MyNeuronWrapper(nn.Module):
       def __init__(self):
           nn.Module.__init__(self)
           self.model = MyActualModel()  # WRONG — parallel_state not active
   ```

   If using an existing NxDI port (e.g. `NeuronQwen3VLTextForCausalLM`), layer construction already happens inside `NeuronBaseModel.init_model()` which is called by ModelBuilder correctly. Subclassing and overriding only `convert_hf_to_neuron_state_dict` is correct — do not rewrite the model.

4. **Eliminate dynamic ops from traced subgraphs.**

   **Dynamic constant pre-compile checklist — run before submitting any block:**
   Does `forward()` or `load_module()` create any of the following at runtime? Move ALL of them to `__init__` as `register_buffer()`:
   - `torch.arange(...)` — position IDs, frequency bases, RoPE sin/cos
   - `torch.linspace(...)` — timestep sequences
   - `torch.ones(...)` / `torch.zeros(...)` — attention masks, padding masks
   - Math over `torch.arange` — sinusoidal embeddings, timescale computations

   In deep models (16+ layers), each dynamic constant creates one compiler node per layer. The compiler merges identical nodes into a single constant whose debug filename includes all parent node IDs — exceeding the 255-char limit causes `[Errno 36] File name too long`. On shallow models it silently degrades performance. **The fix is always the same: pre-compute in `__init__` as `register_buffer()`.**

4b. **Verify TP is actually active — mandatory before submitting any block targeting tp_degree > 1:**

   ```python
   from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
   from neuronx_distributed.parallel_layers import parallel_state

   parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree)
   block = MyNeuronWrapper()
   block.load_module()

   has_parallel = any(isinstance(m, ColumnParallelLinear) for m in block.model.modules())
   assert has_parallel, (
       "TP FAILED: No ColumnParallelLinear found. Layers were constructed before "
       "parallel_state was active. Move construction to load_module()."
   )
   print("TP verification PASSED")
   ```

   **If this assertion fails, fix load_module() before proceeding. Do not compile.**

5. **Verify attention implementation compatibility.** HuggingFace models default to SDPA which breaks CPU tracing. Unless a specific attention implementation is requested in the prompt (e.g. flash attention, a custom NKI kernel), set `_attn_implementation = "eager"` on any HuggingFace reference module. If a non-eager implementation is explicitly requested, verify it is compatible with CPU tracing before proceeding.

   **Qwen3-VL vision encoder: packed attention removal.** The Qwen3-VL ViT uses `cu_seqlens`-based packed attention with `torch.split(..., lengths.tolist())` — data-dependent shapes incompatible with neuronx-cc. Before tracing, replace with standard full-sequence attention over the complete sequence. For single fixed-size images this is numerically identical. Add `_attn_implementation = "eager"` as well.

   **Qwen3-VL `tie_word_embeddings` crash.** The 4B model has `tie_word_embeddings=true` in config, which causes a crash in SDK 2.28 during weight loading. Fix: copy `embed_tokens.weight` → `lm_head.weight` in `convert_hf_to_neuron_state_dict`:
   ```python
   if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
       state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
   ```
   This can also be applied as a pre-compilation monkey-patch on the model object.

6. **Preserve the forward pass contract.** Ensure the translated block accepts and returns tensors matching the shapes and dtypes specified in the integration contract. Do not change semantics — only change the layer implementations.

7. **Write a unit test.** Instantiate both the original PyTorch block and the translated Neuron block with identical weights. Run a forward pass with identical inputs and assert numerical equivalence within an acceptable tolerance. Start with `atol=1e-3` for BF16. If tests fail due to accumulated numerical drift across many layers (common in deep stacks), relax to `rtol=0.05` and document the relaxation. If sinusoidal or frequency-based positional encodings are present, verify timestep and frequency inputs are passed as Python floats rather than tensors — BF16 rounding of large frequency values can cause catastrophic output decorrelation. **The unit test must use `test_block_correctness` from `tests/block_testing_utils.py` following the calling convention documented in Phase 0.**

8. **Document deviations.** If the source block could not be translated exactly, document the deviation, the workaround applied, and any expected numerical differences. Flag cases that may require a custom NKI kernel.

9. **`load_state_dict` overrides must accept and forward `**kwargs`.** Signature: `(self, state_dict, strict=True, **kwargs)`. `torch_neuronx` passes `assign=True` internally — not forwarding it causes a `TypeError` that, if swallowed, produces a silent missing-NEFF failure with no error message.

10. **Never wrap compile calls in bare exception handlers.** No `try/except Exception: pass` around `trace()`, `compile()`, or `compile_denoiser()`. Failures must always propagate as exceptions.

**Subagent deliverables:** translated block file + passing unit test. Deviations in inline comments only. No README, summary, or status files.

### TP Verification Gate — mandatory before Phase 3

After all subagents complete, run the TP verification check (item 4b) on every block targeting tp_degree > 1. Record results in STATUS.md. If any block fails, fix it before proceeding.

### Auditing Subagent Test Files (Anti-Cheat Check)

After each subagent returns, **read the generated test file and verify it is not cheating**. A subagent cheats when it defines or imports the PyTorch reference class from a file it wrote itself. This produces a circular test that always passes regardless of correctness.

**Check for these red flags:**
1. **Local `pytorch_block.py` exists** — read it and confirm it is not a copy of any class in the translated block file.
2. **Import from a local file** — the test imports the reference class from inside the workspace directory. It must point to the original source path outside the workspace.
3. **Reference class shares code with translated block** — re-uses helpers or logic defined in the translated block file.

**If any red flag is detected, relaunch the subagent** with:
> "IMPORTANT: Your previous test was rejected because it imported the PyTorch reference class from a file you wrote yourself. You MUST import `[ClassName]` directly from its original source at `[original_source_path]`. Do NOT create a `pytorch_block.py` file."

### Step — Update STATUS.md

```markdown
## Phase 2 — Complete
- Blocks translated: [list with file names]
- All unit tests passing: yes/no
- TP verification: [PASSED/FAILED per block — must all be PASSED]
- Deviations flagged: [list]
- Dynamic constants moved to register_buffer: [list or none]
```

**Do not stop here. Proceed to Phase 3 immediately without prompting the user.**

---

## Phase 3: Scaffolding and Integration
*See [scaffolding & integration guide](reference/scaffolding_integration.md) for detailed API usage.*

The orchestrating agent collects all subagent deliverables and assembles the complete model. This phase is sequential.

1. **Import translated blocks using `importlib`.** Do not use `sys.path` manipulation — it causes silent module name collisions when multiple blocks share similar file structures. Use unique aliases:

    ```python
    import importlib.util

    def load_block(path, alias):
        spec = importlib.util.spec_from_file_location(alias, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    attn_mod = load_block("attention_block.py", "attention_block")
    mlp_mod  = load_block("mlp_block.py",       "mlp_block")
    ```

2. **Define `NeuronConfig` and `InferenceConfig`.** Implement both config classes using the attribute inventory from Phase 1. Wire `get_neuron_config_cls` to return the `NeuronConfig` subclass. All constants must come from `config_constants.py`.

3. **Assemble `NeuronBaseModel`.** Implement `setup_attr_for_model` and `init_model` using the translated block classes from Phase 2. Ensure all required attributes (`tp_degree`, `hidden_size`, `buckets`, etc.) are set correctly from config.

4. **Define the application head.** Subclass the appropriate task head (e.g., `NeuronBaseForCausalLM`). Set `_model_cls` and wire `get_config_cls`. Leave `convert_hf_to_neuron_state_dict` as a pass-through placeholder — it will be implemented in Phase 4:

    ```python
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        return state_dict  # placeholder — implemented in Phase 4
    ```

5. **Resolve any deviations flagged in Phase 2.** If subagents reported blocks requiring NKI kernels or workarounds, address them now and re-run affected unit tests.

6. **Confirm TP compile path.** For NxDI `NeuronBaseModel` subclasses, `parallel_state` is initialized by `ModelBuilder` before `init_model()` — TP is correct if layer construction is inside `init_model()`. For action heads, `NeuronActionHeadBase.compile_denoiser()` uses `ModelBuilder` — TP is correct if layer construction is inside `load_module()`. Raw `torch_neuronx.trace()` does NOT initialize `parallel_state` — any subgraph compiled this way is TP=1. Document which subgraphs use true TP in notes.md.

### Step — Update STATUS.md

```markdown
## Phase 3 — Complete
- Model assembled: yes
- Config classes: [NeuronConfig subclass, InferenceConfig subclass]
- Deviations resolved: [list or none]
- TP confirmed: [which subgraphs use true TP via ModelBuilder]
```

**Do not stop here. Proceed to Phase 4 immediately without prompting the user.**

---

## Phase 4: Weight Mapping
*This phase requires the assembled Neuron model from Phase 3. See [weight mapping guide](reference/weight_mapping.md) for detailed instructions.*

NxDI models load weights through `convert_hf_to_neuron_state_dict`. Now that the Neuron model is fully assembled, its `state_dict()` can be inspected directly to drive the key mapping. Dispatch a general-purpose agent to implement the weight mapping.

1. **Diff the state dict keys.** Instantiate the Neuron model on CPU (no compilation). Load the HF checkpoint. Print keys present in one but not the other to find every rename, fusion, and missing metadata tensor that the conversion function must produce.

2. **Cross-check shapes against `config_constants.py`.** For every tensor in the HF checkpoint, verify its shape matches the constant recorded in `config_constants.py`. If any shape differs, update `config_constants.py` immediately and flag all Phase 2 blocks that used the affected constant for re-testing.

3. **Implement `convert_hf_to_neuron_state_dict`.** Replace the placeholder with the real implementation. For each discrepancy found in the diff: rename keys, fuse weights (e.g. Q/K/V → Wqkv), apply any required transformations (transpose, scale fusion), and inject rank metadata tensors. Do not manually shard weights — the framework handles sharding at load time.

4. **Validate conversion.** Assert that the converted state dict contains no missing keys relative to the Neuron model and that all tensor shapes match. Then load the weights into the Neuron model and verify forward-pass numerical equivalence against the original HF model (pre-compilation, on CPU) within tolerance.

**Subagent deliverables:** the implemented conversion function and a passing validation script.

### Step — Update STATUS.md

```markdown
## Phase 4 — Complete
- Weight mapping implemented: yes
- Validation passing: yes
- Shape mismatches found and resolved: [list or none]
```

**Do not stop here. Proceed to Phase 5 immediately without prompting the user.**

---

## Phase 5: Compilation, Correctness Validation, and Benchmark

*This phase runs on Trainium hardware. The port is NOT complete until this phase finishes. Do not report completion after Phase 4.*

### Step 1 — Pre-compile checklist (run before touching neuronx-cc)

**A. TP layer verification — mandatory for every subgraph targeting tp_degree > 1:**
```python
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
has_parallel = any(isinstance(m, ColumnParallelLinear) for m in model.modules())
assert has_parallel, "TP FAILED — fix load_module() before compiling"
print("TP check PASSED")
```
If this fails, go back to Phase 2. Do not compile.

**B. CPU forward pass — catches errors before multi-minute neuronx-cc runs:**
```python
model.eval()
with torch.no_grad():
    output = model(*example_inputs)
assert output is not None and not torch.isnan(output).any()
print("CPU forward pass PASSED")
```

**C. Skip-if-compiled check:**
```python
if os.path.exists(save_path + "model.pt"):
    print(f"Already compiled, skipping: {save_path}")
else:
    model.compile(save_path)
```

**D. No compile call is wrapped in try/except.**

### Step 2 — Compile all subgraphs

Compile in dependency order. For each subgraph:
1. Call `model.compile(save_path)` for NxDI models or `compile_denoiser(save_path)` for action heads.
2. **Do NOT call raw `torch_neuronx.trace()` at this level.** Raw trace skips `parallel_state` initialization. It is only acceptable inside `ModelWrapper.load_module()` registered with `ModelBuilder`.
3. Verify output file exists and size > 0.1 MB.
4. Monitor background jobs: `ps aux | grep neuronx-cc` after 2 minutes. Dead process = silent failure.

Compiler flag reference:

| Flag | Effect | When to use |
|------|--------|-------------|
| `-O1` | Standard optimization, no model-type overrides | **Default for all subgraphs** (LLM, VLM, DiT, action heads) |
| `--model-type=transformer` | Replaces softmax with custom NKI kernel | **Never use on DiT/denoising models.** Causes cos_sim=0.916, 37% error per step, blank output. Safe only for standard causal LM backbones that are not DiT. |
| `--model-type=unet-inference` | Optimized for conv/ViT patterns | Vision encoders (SigLIP2, ViT). +6% over transformer for vision encoders. |
| `--auto-cast=matmult` | BF16 matmuls, FP32 accumulators | Vision encoders. ~50% NEFF size reduction, 99.999% accuracy maintained. |
| `--auto-cast=none` | No dtype casting | DiT and action head subgraphs — preserve BF16 throughout. |
| `--optlevel 3` | Maximum optimization | Vision encoders with `--auto-cast=matmult`. |

Compiler error fixes:
- `[Errno 36] File name too long` → dynamic constants in `forward()`. Move `torch.arange`/`torch.ones`/`torch.zeros` to `register_buffer()`. Always the fix.
- `attn_kernel_enabled` → add `NeuronConfig(attn_kernel_enabled=False)` and retry. **Check hardware first** — trn2 at tp=2 requires kernels on, trn1 requires kernels off. See Phase 1 ISA kernel matrix.
- `TypeError: unexpected keyword argument 'assign'` → `load_state_dict` missing `**kwargs`.
- `Expected weight tensors for N ranks. Received 1` → layers constructed in `__init__` not `load_module()`. Fix and recompile.
- `NCC_IBIR039` → bucket size below minimum. Set minimum bucket to 512 (e.g. `buckets=[512, 1024, 2048]`).
- `NCC_EVRF007` → subgraph exceeds 5M compiler instructions. Split at block boundaries. See Phase 1 note.

**Vision encoder compilation pattern.** Vision encoders (SigLIP2, Qwen3-VL ViT) use raw `torch_neuronx.trace()` at TP=1, not NxDI ModelBuilder. See [reference/patterns/vit_compilation.md](reference/patterns/vit_compilation.md) for the full pattern. Expected: 70× speedup over CPU for SigLIP2-giant at batch=1.

**HBM process lifetime.** The Neuron runtime holds NeuronCore allocations for the process lifetime — deleting a model in Python does NOT release HBM. Load subgraphs in the correct order. Use `NEURON_RT_VISIBLE_CORES` to pin components to specific cores (e.g. text encoder on core 0, transformer on cores 1-3). Use a subprocess for components that need a clean core allocation after a previous model has been loaded (e.g. text encoders before a large transformer). See [examples/flux1_lite_8b_trn2.ipynb] for a worked subprocess pattern.

**VLM preprocessing: image resize to patch grid.** Client code must resize images to align with `patch_size × spatial_merge_size` before passing to the vision encoder. For Qwen3-VL this is 56 px (patch_size=14, spatial_merge=4). Arbitrary input sizes cause shape errors inside the compiled graph. Validate input dimensions in the preprocessing wrapper and raise a clear error before the NEFF call.

**Deepstack intermediate features.** Some ViT configurations expose intermediate layer features (e.g. layers 5, 11, 17) in the forward output signature. Verify whether these are consumed at inference time before including them in the compiled graph. Unused outputs add NEFF size and latency for no benefit. Check the calling code — if `run_vit_cpu` or similar preprocessing is still mentioned, clarify exactly what enters the NEFF vs stays on CPU.

### Step 3 — NEFF correctness validation (mandatory — no exceptions)

For EVERY compiled NEFF, validate against HF CPU reference before benchmarking. A NEFF that cannot be validated is broken.

```python
def validate_neff(neff_model, hf_model, example_inputs, name, atol=0.1, cos_threshold=0.99):
    hf_model.eval()
    with torch.no_grad():
        hf_out = hf_model(*example_inputs).float()
        neff_out = neff_model(*example_inputs).float()

    mean_diff = (hf_out - neff_out).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        hf_out.flatten(), neff_out.flatten(), dim=0
    ).item()

    print(f"{name}: mean_diff={mean_diff:.4f}  cos_sim={cos_sim:.6f}")

    assert mean_diff < atol, (
        f"{name} FAILED: mean_diff={mean_diff:.4f} > {atol}. "
        f"NOTE: mean_diff > 1.0 means weights were NOT loaded before trace. "
        f"Fix load_module() to load state dict before returning the model."
    )
    assert cos_sim > cos_threshold, (
        f"{name} FAILED: cos_sim={cos_sim:.6f} < {cos_threshold}"
    )
    print(f"{name}: PASSED")
```

Thresholds:
- Vision encoder: `atol=0.1, cos_threshold=0.999`
- VLM backbone: `atol=0.1, cos_threshold=0.997`
- Action head (single step): `atol=0.15, cos_threshold=0.999`

**If mean_diff > 1.0:** weights not loaded before trace. Fix `load_module()`. Do not proceed.
**Do not benchmark until all NEFFs pass this validation.**

### Step 3b — Open-loop evaluation (VLA models — MANDATORY hard gate)

**This is not optional.** Phase 5 cannot be marked complete without open-loop evaluation results. NEFF correctness validation (Step 3) verifies single-step numerical agreement; open-loop evaluation verifies that errors do not compound across the full denoising loop. A port with passing Step 3 results can still fail open-loop if there is a dtype cast, attention bug, or noise schedule mismatch accumulating across steps.

```bash
python <model_eval_script> \
    --dataset-path <demo_data_path> \
    --embodiment-tag <embodiment> \
    --model-path <compiled_model_path> \
    --traj-ids 0 1 2
```

Also run with HF reference model on CPU. Neuron action MSE must be within 10% of HF reference MSE. If significantly higher, the port has a correctness problem — do not proceed to benchmark. Common causes: `--model-type=transformer` on a DiT (fix: use `-O1`), dtype cast accumulation (fix: verify BF16 throughout), wrong noise schedule (fix: verify timestep sequence matches reference).

### Step 4 — Benchmark

```python
import time, statistics
timings = []
for i in range(70):
    start = time.perf_counter()
    output = model(*inputs)
    elapsed = (time.perf_counter() - start) * 1000
    if i >= 20:
        timings.append(elapsed)

sorted_t = sorted(timings)
print(f"mean={statistics.mean(timings):.1f}ms  "
      f"median={statistics.median(timings):.1f}ms  "
      f"p95={sorted_t[int(0.95*len(sorted_t))]:.1f}ms  "
      f"std={statistics.stdev(timings):.1f}ms  "
      f"throughput={1000/statistics.mean(timings):.2f} inf/sec")
```

### Step 5 — Update STATUS.md

```markdown
## Phase 5 — Complete
- TP verification: [PASSED/FAILED per subgraph]
- NEFFs compiled: yes
- NEFF correctness: [mean_diff and cos_sim per subgraph — must all PASS]
- Open-loop MSE vs HF: [value — VLA only]
- Benchmark: [per-subgraph and end-to-end mean/p95]
```

**Do not stop here. Proceed to Phase 6 immediately without prompting the user.**

---

## Phase 6: Packaging and Usage Instructions

*The port is NOT complete until this phase finishes.*

### Step 1 — Write `run_inference.py`

Self-contained script at the output root. Must:
1. Load all compiled NEFFs and real checkpoint weights
2. Expose a clean API:
   - VLA: `generate_actions(image, instruction: str, num_steps: int) -> np.ndarray`
   - VLM: `generate(image, prompt: str, max_tokens: int) -> str`
   - LLM: `generate(prompt: str, max_tokens: int) -> str`
3. Include `load_model()` importable by `benchmark.py`
4. Include `if __name__ == "__main__":` block with dummy inputs that runs on Trainium

Verify: `python run_inference.py` runs without error and prints non-degenerate output.

### Step 2 — Write `benchmark.py`

```python
from run_inference import load_model
import time, statistics

model = load_model()
timings = []
for i in range(70):
    start = time.perf_counter()
    # run inference
    elapsed = (time.perf_counter() - start) * 1000
    if i >= 20:
        timings.append(elapsed)

sorted_t = sorted(timings)
print(f"mean={statistics.mean(timings):.1f}ms  "
      f"p95={sorted_t[int(0.95*len(sorted_t))]:.1f}ms  "
      f"throughput={1000/statistics.mean(timings):.2f} inf/sec")
```

Verify: `python benchmark.py` prints latency numbers without error.

### Step 3 — Write `README.md`

Replace any prior README.md with one matching this structure exactly:

1. **Header** — one-line model description + validation date
2. **Hardware Requirements** — instance type, NeuronCore count, HBM, Neuron SDK version pin, AMI
3. **Environment Setup** — single command to activate the venv
4. **Architecture Overview** — ASCII diagram showing the full pipeline with tensor shapes at each subgraph boundary
5. **Compiled NEFFs table** — columns: NEFF name, file size, TP degree, input shape(s), output shape(s)
6. **Compile from Scratch** — exact commands with `--only` flags, `--force` flag, compilation time estimates for warm cache and cold cache
7. **Run Inference** — Python API examples showing both Option A (load pre-compiled) and Option B (compile then run)
8. **Benchmark** — per-subgraph table with mean latency and P95; end-to-end pipeline latency
9. **Validate Correctness** — cos_sim table per subgraph; open-loop eval results (MSE vs HF reference)
10. **Compiler Flags** — explain WHY each flag was chosen, not just what was used (include the DiT/ViT distinction and the `--model-type=transformer` warning)
11. **TP Design Rationale** — head count math showing why the chosen tp_degree was selected
12. **Known Deviations** — table of deviations with cos_sim impact per deviation
13. **Directory Structure** — tree of the compiled artifact directory

### Step 4 — Write `demo.ipynb`

Produce a self-contained Jupyter notebook with these sections as markdown headers, each followed by runnable code cells. The notebook must run top-to-bottom on the Trainium instance after NEFFs are compiled. No placeholder cells.

1. **Setup** — imports, path constants, `torch_neuronx` version check
2. **Architecture Overview** — markdown only, ASCII diagram matching the README
3. **Load Checkpoint and Inspect Weights** — load safetensors, print weight groups and key tensor shapes (confirm shapes match `config_constants.py`)
4. **Compile Subgraphs** — skip-if-compiled check for each subgraph, then compile call; print estimated time
5. **Load Compiled NEFFs** — `load_model()` call imported from `run_inference.py`
6. **Sample Inference with Dummy Inputs** — full pipeline run with timing, shape assertion on output, NaN check
7. **Benchmark** — warmup loop (20 iters discarded), measure loop (50 iters), numpy stats (mean, median, P95, std), print formatted table; include ViT CPU vs NEFF latency comparison if applicable
8. **Correctness Validation** — `subprocess.run` call to `validate_neffs.py` and `open_loop_eval.py`, filter and display output

### Step 5 — Final verification

```bash
python run_inference.py
python benchmark.py
jupyter nbconvert --to notebook --execute demo.ipynb
```

Update STATUS.md:

```markdown
## Phase 6 — Complete
- run_inference.py: verified working
- benchmark.py: verified working
- README.md: written (all 13 sections present)
- demo.ipynb: all cells execute without error
- Final benchmark: [end-to-end mean/p95]
```

**The port is complete when Phase 6 STATUS is written. Do not report completion before this.**