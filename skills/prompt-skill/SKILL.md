---
name: generate-prompt
description: Generate a prompt for translating PyTorch models to AWS Trainium
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
- LLM Inference Features: NxD Inference provides support for various LLM inference features like KV Cache, Multi-Head Attention (MHA), Grouped Query Attention (GQA), Flash Attention, Quantization, MoE, Continuous Batching and Speculative Decoding enabling high performance inference.
- Modular Design: Inference features in NxDI like KV Caching are implemented with a modular design, allowing developers to easily incorporate them into new models or customize and extend them.
- Distributed Strategies: NxD Inference enables distributing inference workload of large models across multiple NeuronCores in a single instance using Tensor parallelism and Sequence Parallelism.
- Support for NKI Kernels: NxD Inference provides support for integrating custom NKI kernels on Trainium and Inferentia instances.

# What a Prompt Needs/Workflow
This guide will outline the prompting process and how to go from a model to a prompt.

## User Input
The user should provide a valid PyTorch model and its path, the AWS Trainium instance they are on and venv if in one, the value of tp_degree they want (if they want a specific one), and optionally any constraints they know that would help with the prompt.

Example (Formatting Flexible):
Port Qwen3 VL at Qwen/Qwen3-VL (HuggingFace) in the environment source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate with tp_degree=8. Hard constraint: port vision and text separately.

If the user does not provide a model, or provides an inaccessible model, prompt them for more information until you have a way to access the model. If they do not provide an environment, attempt to go without it, and if this is not possible prompt them for an environment. If they do not provide the tp_degree, the agent will determine the most suitable value during Phase 1. If they do not provide constraints, proceed unconstrained.

## Attaining Information
Your objective is to use the information given to create the most useful prompt for the translate model skill. First, read the skill at reference/translate-reference.md and get familiar with its architecture and how it approaches the port. Focus especially on what the translation needs to complete the port and how this is best communicated to the translator.

### Phase 1: File & Environment Search

*IF* the user has provided a local path, or there is an accessible path through the environment to information on the model, dispatch a subagent to go through these files and get familiarized with the model. The subagent must identify:

1. **Architecture inventory** — every major subgraph present (e.g. vision encoder, LLM backbone, action head) with class names and file paths for each
2. **Model type classification** — classify as one of: text-only LLM, VLM, VLA, or other. For VLAs, further classify the action head type: flow matching (GR00T/SmolVLA-style), DDPM-style (π0-style), discrete tokens (OpenVLA-style), regression MLP, ACT/CVAE, or unknown
3. **Per-subgraph porting notes** — for each subgraph: what it does, whether any user constraints apply to it, and any ops or patterns likely to be problematic for NxDI (e.g. dynamic control flow, non-standard attention, custom positional encodings)
4. **tp_degree recommendation** — if the user did not provide one, recommend a value based on model size, number of NeuronCores on the target instance, and standard NxDI conventions. Include a one-line justification
5. **Tensor boundary map** — for multi-subgraph models, list every tensor that crosses a subgraph boundary with its shape and dtype

If no path was provided or the path is not useful, fall back to prompting the user for more information before proceeding.

**Deliverable** — a structured note covering all five points above. The orchestrator passes this directly to Phase 2 and Phase 3. Do not produce README, summary, or documentation files.

### Phase 2: Web Search

Receives the Phase 1 deliverable as input. Dispatch a subagent to find information about the model using the web, using the Phase 1 architecture inventory to sharpen queries — search for specific class names, config fields, and flagged ops rather than just the model name. Look through GitHub repos, HuggingFace model cards, NVIDIA/vendor documentation, and related technical writeups. The subagent must find:

1. **Architecture details** — attention variant, positional encoding type, any non-standard ops, particularly for anything Phase 1 flagged as uncertain or problematic
2. **Known NxDI compatibility issues** — search for the model name + "neuron" or "trainium" to find existing tracing failures, workarounds, or community ports
3. **HF config schema** — what fields the `PretrainedConfig` actually contains, since this drives `get_required_attributes()`
4. **tp_degree conventions** — if not already determined in Phase 1, search for community benchmarks or official recommendations for this model size on Trainium

**Deliverable** — a structured note covering all four points above. The orchestrator passes this directly to Phase 3. Do not produce README, summary, or documentation files.

### Phase 3: Prompt Generation

Receives both Phase 1 and Phase 2 deliverables as input. Analyze both notes and synthesize into a single well-written translator prompt. The output path should be derived from the model name (e.g. `~/modelname_port/`) unless the user specified one.

The prompt must contain the following sections in order, populated from the Phase 1 and 2 deliverables:

1. **Header** — one-line task description, output path, skill read instructions (SKILL.md first, then the appropriate reference docs based on model type: `vlm_translation.md` for VLMs, `action_head_translation.md` for VLAs, neither for text-only)

2. **Target model and environment** — model path/HF identifier, environment activation command

3. **Hard requirements** — any user-provided constraints stated explicitly as HARD REQUIREMENT blocks. If the user provided none, omit this section entirely — do not invent constraints

4. **Architecture classification guidance** — populated from Phase 1 and 2 findings: subgraph list, which NxDI reference model to use for each subgraph, any flagged ops and recommended workarounds. Route each subgraph to the correct NxDI path:
   - Text-only LLM → causal LM path
   - VLM backbone → `vlm_translation.md` reference pattern (Pixtral/MLlama/Llama4)
   - VLA action head → action head type classification and appropriate `action_head_translation.md` branch
   - Check whether NxDI already ships a port of this model and call it out explicitly if so

5. **Model-specific gotchas** — populated from Phase 1 and 2: non-standard ops, positional encoding variants, shape pinning requirements, known NxDI compatibility issues found in web search. Include tensor boundary shapes from the Phase 1 boundary map

6. **Hardware target** — instance type, NeuronCore count, memory, tp_degree with justification, batch size, latency vs throughput priority

7. **Ordered deliverables** — one deliverable per subgraph following the pattern: CPU block unit tests → CPU end-to-end parity → compiled NEFF. Add an integration smoke test after all subgraphs compile. End with a notes.md deliverable documenting bucket sizes, tp_degree, compiler flags, and deviations. Each deliverable must be a hard gate — do not proceed to the next until the current one passes.

**Deliverable** — a single complete prompt ready to hand to the translator. No preamble, no explanation, just the prompt.