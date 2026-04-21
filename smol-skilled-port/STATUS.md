# Port Status

## Phase 0 — Complete (2026-04-19)
- Environment verified: NxDI OK, torch_neuronx OK, lerobot 0.5.1 OK
- Checkpoint: lerobot/smolvla_base → /home/ubuntu/smol-port-skilled/checkpoint/
- config_constants.py written: yes
- Test harness: tests/block_testing_utils.py (copied from skills/scripts/)

### Test harness calling convention (from block_testing_utils.py)
- Function: `test_block_correctness(neuron_block_class, pytorch_block_class, weight_mapping, ...)`
- Harness casts model to bfloat16 and uses `torch.nn.init.normal_(param, mean=0.0, std=0.02)`
- Reference block called as: `reference_block(reference_inputs[0][0])` — single-arg
- Neuron block wrapped in `_BlockWrapper` for XLA tracing (extracts `.hidden_states` or `[0]` from tuple outputs)
- Config auto-injected into neuron_init_kwargs as `{'config': config, **neuron_init_kwargs}`
- `NeuronConfig` must have `on_cpu=True` for CPU tests to avoid TKG initialization
- Action head tests: `test_action_head_correctness()` and `test_denoising_loop_correctness()`

### Architecture constants verified
- VLM: 16 layers, hidden=960, heads=15/5 (GQA), intermediate=2560
- Vision: 12 layers, hidden=768, patch=16, image=512, 64 tokens/camera after pixel-shuffle-4
- Expert: 16 layers, hidden=720, interleaved self/cross every 2 layers
- Connector: Linear(12288→960) where 12288 = 768*16 (pixel-shuffle groups 4×4)
- Action chunk=50, action_dim=32, state_dim=32, num_denoising_steps=10
- Conditioning interface: [B, 113, 5120] (8 cross-attn layers × 2 KV × 320 kv_dim)

### Key architectural insight
SmolVLA runs VLM and expert TOGETHER layer-by-layer (not sequentially). At inference:
1. Prefix pass: VLM only (fills KV cache for all 16 layers)
2. Denoising loop: Expert only, cross-attends into cached VLM KVs at odd layers
The "L/2 exit" in the task brief refers to the VLM using only 16/32 base layers (pre-baked
in checkpoint). No further truncation needed for the port.

## Phase 1 — COMPLETE (2026-04-20)

All 11 deliverables done. CPU parity tests pass for all 3 blocks.
NEFF compilation deferred (requires Trainium hardware).

## Phase 3 — COMPLETE (2026-04-20)

NxDI scaffolding assembled. All 6 smoke tests pass on CPU.

### Deliverables
- modeling_smolvla_neuron.py — full NxDI scaffolding:
  - SmolVLANeuronConfig(NeuronConfig): tp_degree_vision=1, tp_degree_vlm=8, tp_degree_expert=2
  - SmolVLAInferenceConfig(InferenceConfig): 16 required attributes, get_neuron_config_cls()
  - NeuronSmolVLAVisionModel(NeuronBaseModel): wraps NeuronVisionEncoderBlock
  - NeuronSmolVLABackboneModel(NeuronBaseModel): wraps NeuronVLMBackboneBlock
  - NeuronSmolVLAVisionHead(NeuronBaseForCausalLM): _model_cls, get_config_cls, convert placeholder
  - NeuronSmolVLABackboneHead(NeuronBaseForCausalLM): same pattern
  - NeuronSmolVLAActionHead: re-exported; compile_denoiser() overridden with torch_neuronx.trace guard
  - NeuronSmolVLAPolicy: top-level orchestrator (compile → load_weights → run)
- tests/test_phase3_scaffolding.py — 6 tests all PASS ✓
- action_expert_block.py — compile_denoiser() override added (torch_neuronx.trace with CPU fallback)

### convert_hf_to_neuron_state_dict status
Placeholders only. Implemented in Phase 4.

## Phase 4 — COMPLETE (2026-04-20)

Weight mapping implemented and validated. All 4 tests pass with real checkpoint weights.

### Deliverables
- modeling_smolvla_neuron.py — convert_hf_to_neuron_state_dict implemented:
  - NeuronSmolVLAVisionHead: 198/198 keys mapped, forward pass valid ✓
    - model.vlm_with_expert.vlm.model.vision_model.{rest} → encoder.{rest}
    - encoder.layers.{i}.* → encoder.{i}.* (strip .layers. from ModuleList path)
    - model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight → encoder.connector.proj.weight
  - NeuronSmolVLABackboneHead: 145/145 keys mapped, forward pass valid ✓
    - model.vlm_with_expert.vlm.model.text_model.{rest} → backbone.{rest}
    - embed_tokens.weight dropped (prefix pass takes pre-embedded tokens)
  - Action expert: 153/153 keys mapped via get_hf_to_neuron_weight_mapping() ✓
- vlm_backbone_block.py — dtype cast added to forward() (hidden_states.to(weight.dtype))
- tests/test_phase4_weights.py — 4 tests all PASS ✓ (including e2e policy with real weights)

### End-to-end real-weights validation
- Final actions shape: (1, 50, 32), mean=-0.19, std=2.36 — no NaN/Inf ✓

### Remaining: Phase 5 — NEFF compilation on trn1 hardware

### Deliverables
1. vision_encoder_block.py — NeuronVisionEncoderBlock (SigLIP + connector)
2. tests/test_vision_encoder.py — CPU parity: max_diff=9.77e-04 < atol=1e-2 ✓
3. Compiled NEFF: deferred (requires trn1 hardware)
4. vlm_backbone_block.py — NeuronVLMBackboneBlock (SmolLM2 16-layer prefix pass)
5. tests/test_vlm_backbone.py — CPU parity: max_diff=4.88e-04 < atol=5e-2 ✓
6. Compiled NEFF: deferred (requires trn1 hardware)
7. action_expert_block.py — NeuronSmolVLADenoisingWrapper + NeuronSmolVLAActionHead
8. tests/test_action_expert.py — single step: 0.00e+00, denoising loop: 0.00e+00 ✓
9. Compiled NEFF: deferred (requires trn1 hardware)
10. tests/test_integration.py — end-to-end shape + 10-step denoising loop ✓
11. notes.md — architecture docs, compiler flags, conditioning layout ✓
