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
```

Update `STATUS.md` at the end of every phase.

---

## Phase 1: Model Understanding and Planning

The orchestrator must **not** read source files or plan directly. It dispatches a single `plan` subagent that performs source exploration and returns the full Phase 2 execution plan.

### Step 1 — Dispatch a plan agent for exploration + work plan

Launch a `plan` subagent (thoroughness: "very thorough") with a prompt that instructs it to return one self-contained plan covering:

1. **Source model architecture inventory.** Read the model's PyTorch source and HuggingFace config. Identify every major block type present: attention (MHA/GQA/MQA), MLP, MoE routing and expert layers, embedding tables, normalization layers, positional encodings (RoPE, ALiBi, etc.), and any custom ops. Include file paths and class names for each block.

2. **Reference NxDI model.** Identify which reference model best matches the target architecture and return its full file path:
   - **Text MoE**: [Qwen3 MoE](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/qwen3_moe/modeling_qwen3_moe.py)
   - **VLM (scatter integration)**: [Pixtral](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/pixtral/modeling_pixtral.py)
   - **VLM (cross-attention integration)**: [MLlama](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/mllama/modeling_mllama.py)
   - **VLM (chunked vision flow)**: [Llama4](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/llama4/modeling_llama4.py)
   - **VLA (flow matching)**: [SmolVLA](https://github.com/huggingface/lerobot/blob/main/lerobot/common/policies/smolvla/)
   - **Other (dense text)**: [Llama](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/llama/modeling_llama.py)

3. **NxDI compatibility assessment.** For each block, explicitly state whether it maps cleanly to an NxDI primitive. NxDI is designed for standard autoregressive LLM inference — models that deviate from this pattern will require partial or full `torch_neuronx.trace()`. Non-standard patterns to watch for include: denoising or diffusion loops, flow matching, non-autoregressive generation, fused multi-model architectures, custom KV cache layouts, dynamic control flow, and task heads that output non-token tensors (e.g. actions, embeddings, class logits). Document which subgraphs should use NxDI primitives and which should use `torch_neuronx.trace()`. Do not silently fall back — every `trace()` usage must be explicitly justified in the plan.

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

3. **Implement the Neuron block class.** Subclass the appropriate NxDI base (`NeuronAttentionBase`, `NeuronBaseModel`, etc.) per the substitution map from Phase 1, or use `torch_neuronx.trace()` if explicitly specified. Replace all standard PyTorch layers with their Neuron parallel equivalents per the substitution map.

4. **Eliminate dynamic ops from traced subgraphs.** Any op that produces a dynamic shape or depends on a Python-level value at trace time will cause a shape error. Common offenders: `torch.linspace`, `torch.arange`, `torch.randint`, `torch.multinomial`, adaptive solvers, and any op whose output shape depends on input values rather than input shapes. Move these to `__init__` as `register_buffer` (for fixed tensors) or to Python code outside the compiled subgraph (for loop counters, timesteps, etc.).

5. **Verify attention implementation compatibility.** HuggingFace models default to SDPA which breaks CPU tracing. Unless a specific attention implementation is requested in the prompt (e.g. flash attention, a custom NKI kernel), set `_attn_implementation = "eager"` on any HuggingFace reference module. If a non-eager implementation is explicitly requested, verify it is compatible with CPU tracing before proceeding.

6. **Preserve the forward pass contract.** Ensure the translated block accepts and returns tensors matching the shapes and dtypes specified in the integration contract. Do not change semantics — only change the layer implementations.

7. **Write a unit test.** Instantiate both the original PyTorch block and the translated Neuron block with identical weights. Run a forward pass with identical inputs and assert numerical equivalence within an acceptable tolerance. Start with `atol=1e-3` for BF16. If tests fail due to accumulated numerical drift across many layers (common in deep stacks), relax to `rtol=0.05` and document the relaxation. If sinusoidal or frequency-based positional encodings are present, verify timestep and frequency inputs are passed as Python floats rather than tensors — BF16 rounding of large frequency values can cause catastrophic output decorrelation. **The unit test must use `test_block_correctness` from `tests/block_testing_utils.py` following the calling convention documented in Phase 0.**

8. **Document deviations.** If the source block could not be translated exactly (e.g., an unsupported op), document the deviation, the workaround applied, and any expected numerical differences. Flag cases that may require a custom NKI kernel.

**Subagent deliverables:** a translated block class file (named after the block) and a passing unit test. Deviations go in inline comments only. Do NOT produce README, summary, status, or documentation files.

### Auditing Subagent Test Files (Anti-Cheat Check)

After each subagent returns (or while it is operating, if you can observe its workspace), **read the generated test file and verify it is not cheating**. A subagent cheats when it defines or imports the PyTorch reference class from a file it wrote itself, rather than from the original source. This produces a circular test that always passes regardless of correctness.

**Check for these red flags:**

1. **Local `pytorch_block.py` exists** — If a `pytorch_block.py` file is present in the workspace, the subagent almost certainly wrote the reference class itself. Read it and confirm it is not a copy or paraphrase of any class in the translated block file.
2. **Import from a local file** — The test imports `PyTorchBlock` (or any reference class) from a file inside the workspace directory. The import must point to the original source path outside the workspace.
3. **Reference class shares code with translated block** — The reference class re-uses helpers, constants, or logic defined in the translated block file.

**If any red flag is detected, relaunch the subagent** with an explicit instruction prepended to its prompt:

> "IMPORTANT: Your previous test was rejected because it imported the PyTorch reference class from a file you wrote yourself. You MUST import `[ClassName]` directly from its original source at `[original_source_path]`. Do NOT create a `pytorch_block.py` file. Do NOT copy or rewrite the reference class."

Re-run the audit after the subagent returns again. Only accept a result once the test file imports the reference class from the unmodified original source.

### Step — Update STATUS.md

After all subagents complete:

```markdown
## Phase 2 — Complete
- Blocks translated: [list with file names]
- All unit tests passing: yes/no
- Deviations flagged: [list]
- NKI kernels required: [list or none]
```

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

### Step — Update STATUS.md

```markdown
## Phase 3 — Complete
- Model assembled: yes
- Config classes: [NeuronConfig subclass, InferenceConfig subclass]
- Deviations resolved: [list or none]
```

---

## Phase 4: Weight Mapping
*This phase requires the assembled Neuron model from Phase 3. See [weight mapping guide](reference/weight_mapping.md) for detailed instructions.*

NxDI models load weights through `convert_hf_to_neuron_state_dict`. Now that the Neuron model is fully assembled, its `state_dict()` can be inspected directly to drive the key mapping. Dispatch a general-purpose agent to implement the weight mapping.

1. **Diff the state dict keys.** Instantiate the Neuron model on CPU (no compilation). Load the HF checkpoint. Print keys present in one but not the other to find every rename, fusion, and missing metadata tensor that the conversion function must produce.

2. **Cross-check shapes against `config_constants.py`.** For every tensor in the HF checkpoint, verify its shape matches the constant recorded in `config_constants.py`. If any shape differs, update `config_constants.py` immediately and flag all Phase 2 blocks that used the affected constant for re-testing.

3. **Implement `convert_hf_to_neuron_state_dict`.** Replace the placeholder with the real implementation. For each discrepancy found in the diff: rename keys, fuse weights (e.g. Q/K/V → Wqkv), apply any required transformations (transpose, scale fusion), and inject rank metadata tensors. Do not manually shard weights — the framework handles sharding at load time.

4. **Validate conversion.** Assert that the converted state dict contains no missing keys relative to the Neuron model and that all tensor shapes match. Then load the weights into the Neuron model and verify forward-pass numerical equivalence against the original HF model (pre-compilation, on CPU) within tolerance.

**Subagent deliverables:** the implemented conversion function replacing the placeholder, and a passing validation script.

### Step — Update STATUS.md

```markdown
## Phase 4 — Complete
- Weight mapping implemented: yes
- Validation passing: yes
- Any shape mismatches found and resolved: [list or none]
```