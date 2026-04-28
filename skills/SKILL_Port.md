---
name: port-model
description: End-to-end VLA/VLM/LLM port to AWS Trainium. Give it a model, it produces a compiled, validated, benchmarked, packaged deployment. No separate steps required.
---

# Overview

This skill takes a model identifier and produces a working Trainium deployment.
It runs without stopping from model identification through to a verified,
benchmarked, packaged output with usage instructions.

Do not ask the user for clarification unless the model cannot be accessed at all.
Make reasonable decisions and document them in notes.md.

---

# Stage 1: Intelligence Gathering

## Step 1.1 — Parse user input

Extract:
- **Model identifier** — HuggingFace path, local path, or name. Ask if inaccessible.
- **Instance type** — trn1.2xl, trn1.32xl, trn2.48xl. Ask if not provided. Determines tp_degree ceiling.
- **Environment** — default: `source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`
- **Output directory** — default: `~/modelname_port/`
- **Hard constraints** — user-specified requirements only. Do not invent.
- **tp_degree** — determine in Step 1.3 if not provided.

## Step 1.2 — Check installed NxDI package first

Before web search or file reading, run:
```bash
ls /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/
```

Note every directory. If the target model's architecture is already there
(e.g. `qwen3_vl/` for a Qwen3-VL backbone), record the exact class names.
These will be subclassed directly — layer construction inside `NeuronBaseModel.init_model()`
already happens after `parallel_state` is active, making it correct by default.

## Step 1.3 — Architecture discovery

Dispatch a subagent to read model source files and/or fetch from GitHub/HuggingFace.
Return:

1. **Model type** — text LLM / VLM / VLA / other
2. **Subgraph inventory** — for each compiled subgraph:
   - Source class name and file path
   - Closest installed NxDI port (from Step 1.2), or None
   - Whether to subclass existing port (preferred) or write custom blocks
   - Problematic ops (dynamic shapes, custom attention, embodiment-specific layers)
3. **Action head type** (VLA only) — flow matching / DDPM / discrete / regression / ACT / unknown
4. **Tensor boundary map** — every cross-subgraph tensor with shape and dtype
5. **tp_degree per subgraph** — based on model size and instance NeuronCore count:
   - VLM backbone 2-3B on trn1.32xl: tp_degree=8
   - Small subgraphs <500M params: tp_degree=1
   - Action head DiT: tp_degree=1 to start, profile before going higher
6. **Key gotchas** — mRoPE, AdaLN, dynamic layer selection, embodiment MLPs,
   flash attention defaults, select_layer truncation, CategorySpecificMLP

Also dispatch a web search subagent to find architecture details for anything
the file search left uncertain and known NxDI/Neuron compatibility issues.

## Step 1.4 — Generate the translator prompt

Synthesize findings into a complete prompt. The prompt MUST explicitly state
for every subgraph:

1. Whether to subclass an existing NxDI port or write custom blocks
2. That layer construction goes in `load_module()` NOT `__init__()`
3. That ModelBuilder must be used (not raw trace()) for tp_degree > 1
4. The tp_degree and justification
5. Any CPU-only components that must NOT be compiled

Prompt structure:
```
Port [MODEL] to AWS Trainium using NxDI.
Output all files to [OUTPUT_DIR].

Read skills in this order before writing any code:
1. skills/SKILL.md
2. skills/reference/vlm_translation.md        (if VLM or VLA)
3. skills/reference/action_head_translation.md (if VLA)

Target model: [HF_IDENTIFIER]
Environment: source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

[HARD REQUIREMENTS — user-specified only, omit section if none]

EXISTING NXDI PORTS — USE THESE, DO NOT REWRITE
[For each subgraph with a matching installed port:
  - Exact class to subclass (e.g. NeuronQwen3VLTextForCausalLM)
  - The ONE change needed: override convert_hf_to_neuron_state_dict
  - Checkpoint key prefix differences to handle]

SUBGRAPHS REQUIRING CUSTOM BLOCKS
[For each subgraph without an installed port:
  - Source class and file path
  - NxDI primitives to use
  - MANDATORY: layer construction in load_module() not __init__()
  - tp_degree with justification]

CPU-ONLY COMPONENTS — DO NOT COMPILE
[List components like CategorySpecificMLP, noise schedules, timestep sequences
 that must stay on CPU]

KEY GOTCHAS
[Populated from Stage 1: mRoPE pre-compute, AdaLN tensor ops, dynamic constants,
 flash attention override, select_layer truncation, etc.]

HARDWARE TARGET
Instance: [type] — [NeuronCore count, memory]
tp_degree per subgraph: [list]
batch_size: 1, latency priority

OUTPUT STRUCTURE
[OUTPUT_DIR]/
├── skills/
├── compiled/[per subgraph]/
├── weights/
├── config/
├── tests/
├── config_constants.py
├── run_inference.py
├── benchmark.py
├── STATUS.md
└── README.md

DELIVERABLES — strict order, hard gates
[Per subgraph: CPU unit tests → CPU parity → TP verification → NEFF compile → NEFF correctness validation]
[Integration smoke test]
[Open-loop evaluation — VLA only]
[Benchmark]
[run_inference.py verified on Trainium]
[benchmark.py prints numbers]
[README.md written]
```

## Step 1.5 — Proceed to Stage 2

Pass the generated prompt to Stage 2. Do not pause unless the user asked for review.

---

# Stage 2: Execute the Port

The prompt generated in Stage 1 is your task specification. Execute it fully.

Read `[OUTPUT_DIR]/skills/SKILL.md` before writing any code. All skill reference
docs are in `[OUTPUT_DIR]/skills/reference/`. All scripts are in
`[OUTPUT_DIR]/skills/scripts/`. Execute all phases (0 through 6). Do not stop
between phases. Do not ask for user input between phases.

**The port is complete only when ALL of the following are true:**
1. Every NEFF passes correctness validation (mean_diff < 0.1, cos_sim > 0.99 vs HF reference)
2. Open-loop evaluation MSE within 10% of HF reference (VLA models)
3. `python run_inference.py` runs without error on Trainium hardware
4. `python benchmark.py` prints latency numbers
5. README.md is written
6. STATUS.md shows Phase 6 Complete

Do not report completion before all six are true.
