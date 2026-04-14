# Port Status — Qwen3-VL → AWS Trainium (NxDI)

## Phase 0 — Complete
- Environment verified: 2026-04-13
- Hardware: trn1.32xlarge (16 Neuron devices, 32 NeuronCores, 512 GiB device memory)
- NxDI version: neuronx_distributed_inference (installed in /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference)
- Checkpoint inspected: No local checkpoint available (HF hub gated).
  Architecture constants derived from transformers.models.qwen3_vl.configuration_qwen3_vl defaults
  (7B-Instruct variant), cross-referenced against the NxDI qwen3_vl source.
- config_constants.py written: yes
- qwen3_vl NxDI port found: YES — production port exists at
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/
  neuronx_distributed_inference/models/qwen3_vl/
  Files: modeling_qwen3_vl.py, modeling_qwen3_vl_text.py, modeling_qwen3_vl_vision.py, utils/slicing.py
- Job: Integration, testing, and validation (not reimplementation)
- Test harness calling convention: direct pytest-style assertions on forward pass outputs
- Test harness dtype casting: bfloat16 on Neuron; float32 on CPU for comparison

## Phase 1 — Complete
- Reference model: NxDI qwen3_vl (production port, Pixtral-style scatter integration)
- Secondary references: NxDI qwen3_moe (text decoder), NxDI pixtral (vision encoder pattern)
- Architecture: Pixtral-style scatter integration (vision ViT → projector → scatter into text)
- NxDI compatibility: ALL blocks use NxDI primitives (NeuronAttentionBase, Parallel layers, etc.)
- Block partition:
    Vision encoder:
      - NeuronQwen3VLVisionPatchEmbed (Conv3d patch embedding)
      - NeuronQwen3VLVisionAttention (NeuronAttentionBase, 2D rotary pos emb)
      - NeuronQwen3VLVisionMLP (ColumnParallelLinear + RowParallelLinear)
      - NeuronQwen3VLVisionBlock (norm + attn + mlp)
      - NeuronQwen3VLVisionPatchMerger (projector to text hidden size)
      - NeuronQwen3VLVisionModel (full vision subgraph)
      - NeuronQwen3VLVisionModelWrapper (preprocessing + bucketing)
    Text decoder:
      - NeuronQwen3VLRotaryEmbedding (MRoPE with 3D position ids)
      - NeuronQwen3VLAttention (NeuronAttentionBase, GQA, q/k layernorms)
      - NeuronLlamaMLP (reused from llama, SwiGLU)
      - NeuronQwen3VLDecoderLayer (norm + attn + mlp)
      - NeuronQwen3VLTextModel (full text subgraph)
- Any non-standard patterns: deepstack_visual_indexes (intermediate layer feature capture),
  3D MRoPE position ids (shape [3, B, S] not [B, S])

## Phase 2 — Complete (CPU validation)
- All 43 tests pass: 14 vision block, 5 vision E2E, 14 text block, 9 text E2E
- Test run: python -m pytest tests/ -v (all 4 files in one session)
- Key CPU test workarounds:
  - cpu_mode() returns False on trn1 (Neuron HW present) → patched to True for text tests
  - NeuronBaseModel.forward() is inference-serving API (requires sampling_params, KV-cache);
    text E2E tests use manual sub-module chaining: embed_tokens → layers → norm → lm_head
  - NeuronBaseModel.__init__ creates SPMDRank → needs torch.distributed initialized;
    test_text_e2e.py initializes gloo dist (rank=0, world_size=1) at module load time
  - rank_util.rank (SPMDRank buffer) appears in state_dict when parallel state initialized;
    test_weight_loading_from_hf uses strict=False + explicit key validation

## Phase 3 — Scripts written; NEFF compilation pending hardware run

## Deliverables Status
- [x] D1: Vision encoder CPU block unit tests — 14/14 passing
- [x] D2: Vision encoder E2E CPU parity — 5/5 passing
- [x] D3: compile_vision.py written — NeuronQwen3VLForImageEncoding instantiates cleanly
      Run: python compile_vision.py [--checkpoint PATH] [--output compiled/vision]
- [x] D4: Text decoder CPU block unit tests — 14/14 passing
- [x] D5: Text decoder E2E CPU parity — 9/9 passing
- [x] D6: compile_text.py written — NeuronQwen3VLForCausalLM instantiates cleanly
      Run: python compile_text.py [--checkpoint PATH] [--output compiled]
- [x] D7: smoke_test.py written — validates D3/D6 artifacts + loads on Neuron
      Run: python smoke_test.py [--compiled-dir compiled]
- [x] D8: notes.md written — vision buckets, tp_degree, compiler flags, deviations
