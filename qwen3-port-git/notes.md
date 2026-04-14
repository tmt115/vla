# Qwen3-VL → AWS Trainium (NxDI) Port Notes

## Architecture Summary

Qwen3-VL-7B-Instruct uses a **Pixtral-style scatter integration**: the vision
ViT encodes image patches, a patch merger projects them to the text hidden
dimension, and the resulting vision embeddings are scattered into the text
token sequence at positions marked by `image_token_id` (151655).

### Vision Encoder (ViT)
| Parameter | Value |
|---|---|
| depth | 27 transformer blocks |
| hidden_size | 1152 |
| intermediate_size | 4304 |
| num_heads | 16 |
| patch_size | 16 × 16 spatial, 2 temporal |
| spatial_merge_size | 2 (PatchMerger compresses H×W by 2×) |
| out_hidden_size | 3584 (= text hidden_size) |
| num_position_embeddings | 2304 (48 × 48 grid) |

### Text Decoder
| Parameter | Value |
|---|---|
| hidden_size | 3584 |
| num_attention_heads | 28 |
| num_key_value_heads | 4 (GQA) |
| num_hidden_layers | 28 |
| intermediate_size | 18944 |
| head_dim | 128 |
| vocab_size | 151936 |
| rope_type | default, mrope_section=[24,20,20] |
| max_position_embeddings | 32768 |

---

## Compilation Configuration

### tp_degree = 8
Chosen for a **trn1.32xlarge** (16 Neuron devices, 32 NeuronCores).  With
tp_degree=8 each device handles one tensor-parallel slice; 32 NeuronCores /
4 cores-per-device = 8 devices used.

### Vision Buckets: [784, 3136, 12544]
Images are bucketed on the pre-merger patch count (patches before spatial
merge). Three buckets cover practical input resolutions:

| Bucket | Patches | Image resolution | Post-merge tokens |
|---|---|---|---|
| 784 | 1 × 28 × 28 | ~448 × 448 px | 196 |
| 3136 | 1 × 56 × 56 | ~896 × 896 px | 784 |
| 12544 | 1 × 112 × 112 | ~1792 × 1792 px | 3136 |

Bucketing is enabled by setting `enable_bucketing=True` and
`buckets=[784, 3136, 12544]` in `Qwen3VLNeuronConfig` for the vision
sub-model.

### Text Bucket: [8192]
Single bucket covering the full context window.  Setting a single bucket
avoids recompile overhead for the text model; all prompts are padded to 8192.

### Compiler flags (vision and text)
```
--auto-cast=none
--model-type=transformer
--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'
-O1
--internal-max-instruction-limit=15000000
```
`--auto-cast=none` keeps the model in bfloat16 throughout; no dtype
promotion.  `--cc-pipeline-tiling-factor=2` enables collective-compute
overlap for AllReduce in the tensor-parallel attention projection.
`--internal-max-instruction-limit` is raised from the default to handle the
large vision ViT graph.

---

## Deviations from Vanilla HF Qwen3VL

### 1. MRoPE position IDs: [3, B, S] not [B, S]
HF Qwen3VL text model accepts standard 2-D `position_ids` of shape `[B, S]`.
The NxDI port uses **3-D MRoPE** position IDs of shape `[3, B, S]` — one
slice each for the temporal, height, and width axes.  For text-only tokens
all three slices are identical (the standard 1-D sequence position); for
vision tokens the three slices encode the spatio-temporal grid coordinates.

### 2. q\_norm / k\_norm key renaming
HF weights store per-head norms as `q_norm` / `k_norm`; NxDI renames them
to `q_layernorm` / `k_layernorm` to match its `NeuronAttentionBase`
interface.

### 3. QKV projection split → nested path
HF stores separate `q_proj`, `k_proj`, `v_proj`.  NxDI groups them under
`qkv_proj.{q,k,v}_proj`; the `convert_hf_to_neuron_state_dict` helper in
`NeuronQwen3VLForCausalLM` performs this renaming at load time.

### 4. Output projection nested path
HF: `self_attn.o_proj.weight`  
NxDI: `self_attn.o_proj.o_proj.weight`

### 5. pos\_embed loaded at model construction (not at load-weight time)
`NeuronQwen3VLVisionModelWrapper.__init__` loads
`model.visual.pos_embed.weight` (shape [2304, 1152]) from the checkpoint
**during Python model construction**, before `compile()` or `load()` is
called.  This is the only weight needed at compile time; all other weights
are loaded during `load_weights()` at runtime.

### 6. deepstack\_visual\_indexes
The vision encoder can optionally capture intermediate feature maps at
layers [8, 16, 24] (`deepstack_visual_indexes`) and inject them into the
corresponding text decoder layers.  This "deepstack" path is supported by
the NxDI port and the `Qwen3VLInferenceConfig` must carry
`deepstack_visual_indexes` in the vision config so that the text config
receives a copy (set by `Qwen3VLInferenceConfig.__init__`).

### 7. NKI CTE attention kernel disabled (NeuronX 2.24 compiler bug)
`attn_kernel_enabled` defaults to `None`, which the NxDI attention base treats
as "auto" — when `q_len >= 4096` it automatically activates the NKI
`attention_cte` flash-attention kernel.  With `TEXT_SEQ_LEN=8192` this fires
on every CTE pass and triggers a compiler internal error:

```
[NCC_INLA001] checkDMATranspose: DMACopy: transpose only supported for HBM->SB
Source Kernel "nkilib.core.attention.attention_cte.attention_cte", line 2258
```

Fix: set `attn_kernel_enabled=False` in `Qwen3VLNeuronConfig` for the text
sub-model.  This forces the standard XLA attention path.  The vision model
already has this flag force-disabled by
`validate_vision_model_supported_configs()`.

### 8. NeuronLlamaMLP reuse
The text decoder MLP is identical to LLaMA's SwiGLU MLP; NxDI reuses
`NeuronLlamaMLP` from `neuronx_distributed_inference.models.llama`.

---

## CPU Test Workarounds (Phases 1–2)

These are test-infrastructure issues, not production deployment concerns.

### cpu\_mode() patch
On trn1 with Neuron hardware present, `cpu_mode()` returns `False`, which
causes `get_rmsnorm_cls()` to return `CustomRMSNorm` (requires XLA tensors).
Text-model CPU tests patch this to `True` at module level so that
`LlamaRMSNorm` is used instead.

### Distributed init for text model
`NeuronBaseModel.__init__` always creates `SPMDRank`, which calls
`get_tensor_model_parallel_size()` — this requires `torch.distributed` and
NxD model-parallel to be initialized even for single-rank CPU tests.  Fixed
by calling `torch.distributed.init_process_group("gloo", rank=0,
world_size=1)` + `initialize_model_parallel(1, skip_collective_init=True)`
at module load time in the text E2E test file.

### NeuronBaseModel.forward() not for unit tests
`NeuronBaseModel.forward()` is the full inference-serving API; it expects
`sampling_params` as a positional argument and requires KV-cache machinery.
Text E2E CPU tests bypass it with a manual sub-module chain:
`embed_tokens → layers → norm → lm_head`.

### rank\_util.rank in state\_dict
`SPMDRank` is registered when parallel state is initialized, adding a
`rank_util.rank` buffer to each attention module's `state_dict`.  The text
block test for weight loading uses `strict=False` and explicitly validates
that all missing keys are `rank_util`-prefixed.
