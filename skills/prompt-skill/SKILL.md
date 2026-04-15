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

**Deliverable** — a structured note covering all five points above. The orchestrator consumes this directly to drive prompt generation in Phase 3. Do not produce README, summary, or documentation files.

### Phase 2: Web Search
Dispatch a subagent to find information about the model to port using the web. Look through github repos pertaining to the model, websites from designers (e.g. HuggingFace, NVIDIA), and related websites to get more information about the model architecture. The subagent must find:

1. **Architecture details** — attention variant, positional encoding type, any non-standard ops, if the HF model card doesn't make it obvious
2. **Known NxDI compatibility issues** — search for the model name + "neuron" or "trainium" to see if anyone has already hit tracing failures or workarounds
3. **HF config schema** — what fields the PretrainedConfig actually contains, since this drives get_required_attributes()
4. **tp_degree conventions** — if model size is ambiguous, search for community benchmarks to confirm the standard recommendation

**Deliverable** — a structured note covering all four points above. The orchestrator consumes this directly to drive prompt generation in Phase 3. Do not produce README, summary, or documentation files.

### Phase 3: Prompt Generation
Firstly, analyze the notes provided during phases 1 and 2. You must orchestrate information into a well-written prompt which provides all the scraped information to the translator. Do not spare any details. In this prompt, you want to advise the agent based on the information you have, giving it guidelines, constraints, ideas, and possibly examples for the port. Use the reading of the translator skill to advise your formatting, do so in a way that will give it the best organization of information.

Prompt Example:

```
Port the Qwen3 VL vision-language model to AWS Trainium using NxDI.
Output all files to ~/qwen_port/.

Follow the translate-model skill: read ~/qwen_port/SKILL.md first,
then read ~/qwen_port/reference/vlm_translation.md before writing any code.

Target model: Qwen/Qwen3-VL (HuggingFace). Use the installed NxDI environment:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

HARD REQUIREMENT — two independent translation units:
Port the vision encoder and the text decoder as completely separate subgraphs.
Each must be independently compiled, saved, and loadable. Do not merge them into
a single traced graph. The interface between them is a single tensor handoff:
vision_embeddings [B, num_vision_tokens, text_hidden_size] passed from the vision
subgraph to the text subgraph at prefill. Complete and validate one subgraph before
starting the other. Recommended order: vision encoder first, then text decoder.

Architecture classification guidance:
Qwen3-VL uses a vision transformer (ViT) with 2D/MRoPE positional encoding and a
projector that scatters vision embeddings into the text token sequence — this is
Pixtral-style scatter integration, not cross-attention. Use the Pixtral reference
as your primary structural guide. The text decoder is Qwen3-architecture (GQA,
SwiGLU MLP, RMSNorm) — use the existing Qwen3 MoE reference at:
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/
  neuronx_distributed_inference/models/qwen3_moe/modeling_qwen3_moe.py
as the text-side reference (dense Qwen3 attention/MLP patterns are identical).

NxDI now ships a production Qwen2-VL and Qwen3-VL port — read it before writing
anything:
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/
  neuronx_distributed_inference/models/qwen2_vl/

Check whether a qwen3_vl directory also exists alongside it. If it does, your job
is integration, testing, and validation — not reimplementation. If it does not,
use the qwen2_vl port as the primary reference and adapt for Qwen3-VL differences
(RMSNorm eps, rope scaling, any attention config changes).

Key things to watch for specific to Qwen3-VL:
- MRoPE (multi-modal RoPE): position ids are 3D [t, h, w] for vision tokens. The
  vision wrapper must build these correctly per bucket.
- Vision token count varies with image resolution. Pin to a fixed set of vision
  buckets matching realistic input sizes for your use case.
- The text decoder is inference-only so disable any training-only paths
  (e.g. gradient checkpointing flags).
- Static shape contract at the vision/text boundary: num_vision_tokens must be
  fixed per vision bucket.

Hardware target: trn1.32xl (32 NeuronCores, 512 GiB device memory).
- tp_degree=8 is a reasonable starting point; adjust based on profiling.
- Single-environment inference, batch size 1, latency priority.
- Vision and text subgraphs compiled and saved separately under
  ~/qwen_port/compiled/vision/ and ~/qwen_port/compiled/text/.

Deliverables in strict order — do not proceed to the next until the current one passes:

1. Vision encoder: passing CPU-side block unit tests for vision attention, vision
   MLP, and the MRoPE position id builder.
2. Vision encoder: passing end-to-end CPU forward parity against HF model
   (image prefill, check vision_embeddings shape and values).
3. Vision encoder: compiled NEFF saved to ~/qwen_port/compiled/vision/.
4. Text decoder: passing CPU-side block unit tests for text attention (GQA),
   MLP, and RMSNorm.
5. Text decoder: passing end-to-end CPU forward parity against HF model
   (text prefill with dummy vision_embeddings, one decode step).
6. Text decoder: compiled NEFF saved to ~/qwen_port/compiled/text/.
7. Integration smoke test: load both compiled subgraphs, run image+text prefill
   followed by one token generation step, verify output is non-degenerate.
8. A short note at ~/qwen_port/notes.md documenting: chosen vision bucket sizes,
   tp_degree, compiler flags used, and any deviations from the HF reference.Port the Qwen3 VL vision-language model to AWS Trainium using NxDI.
Output all files to ~/qwen_port/.
```

**Deliverable** - A well-written highly detailed prompt designed to call the translator for the best possible port.