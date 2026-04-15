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

If no path was provided or the path is not useful, fall back to prompting the user for more information.

**Deliverable** — a structured note covering all five points above. The orchestrator consumes this directly to drive prompt generation in Phase 2. Do not produce README, summary, or documentation files.

If no path was provided or the path is not useful, fall back to prompting the user for more information before proceeding.