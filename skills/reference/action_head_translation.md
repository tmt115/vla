# Handling The Action Head

The action head is the portion of the VLA which takes in the VLM output and uses
it to predict the next action. There are several different architectural approaches,
split broadly into discrete and continuous token prediction. This document covers
how to port each type correctly.

## Part 1: Classification and Branching

Dispatch a subagent to explore the model source with the goal of classifying the
action head type. Read the action head class definition, its forward pass, and the
noise/sampling config if present. Look for the following signals:

**Discrete token indicators:**
- Action values are binned into vocabulary tokens (e.g. OpenVLA-style uniform
  binning mapped to least-used tokenizer entries)
- The action head forward pass appends action tokens to the LM sequence and
  predicts them autoregressively
- Config contains fields like `num_action_bins`, `action_token_begin_idx`, or
  similar

**Continuous token indicators:**
- Action values are real-valued vectors, not token indices
- The forward pass takes a noisy action chunk and a timestep/noise level as input
  and predicts a denoised version
- Config or imports reference a noise scheduler, flow matching, DDPM, or
  diffusion process
- Further classify continuous heads by subtype:
  - **Flow matching** (GR00T N1.6, SmolVLA, π0) — interpolates linearly between
    noise and data, noise schedule is a straight path, typically fewer denoising
    steps. All three confirmed flow matching models.
  - **DDPM-style** — fixed Markov chain with discrete timesteps, longer denoising
    chains. No major current VLA confirmed to use this — flag as unknown if
    encountered and verify against the noise scheduler source before assuming.
  - **Regression MLP** — no denoising loop, directly regresses action from VLM
    features in a single forward pass
  - **ACT/CVAE** — encoder-decoder architecture, produces action chunks via
    latent sampling, no iterative denoising

If you cannot determine the type after reading the source and config, prompt the
user. If the user does not know, treat as unknown and use the fallback path.

Once classification is complete, branch as follows:

1. **Discrete tokens** — Do not follow this document. Return to SKILL.md and
   follow the standard causal LM path. The action tokens are vocabulary tokens
   and the existing NxDI causal LM machinery handles them without modification.

2. **Regression MLP** — Simple case. Port as a standard MLP using
   `ColumnParallelLinear` + `RowParallelLinear`. No denoising loop, no wrapper
   needed. Follow Part 2 for the compiled graph boundary, then skip to Part 4.

3. **Flow matching or DDPM-style** — Continue to Part 2. These require a
   `NeuronDenoisingWrapper` and a CPU-side denoising loop.

4. **ACT/CVAE** — Follow Part 2 for the compiled graph boundary. The encoder
   and decoder compile as separate subgraphs. The latent sampling step stays on
   CPU. Flag any dynamic control flow in the encoder for NKI or workaround.

5. **Unknown** — Fall back to `torch_neuronx.trace()` on the full action head
   as a single subgraph. Document the fallback in notes.md and open a ticket to
   extend this skill with the new type.

**Deliverable** — None. The result of this phase is correct branching behavior
and a classification note passed to subsequent phases.

---

## Part 2: The NeuronDenoisingWrapper

The `NeuronDenoisingWrapper` is the class that owns the single-step denoiser
forward — the part that gets compiled to a NEFF. The N-step denoising loop lives
outside this wrapper entirely, on the CPU side. This boundary is the most
important architectural decision in the action head port and must be correct
before writing any other code.

Both `NeuronDenoisingWrapper` and `NeuronActionHeadBase` are defined in
`scripts/neuron_action_head_base.py` in the skill directory. Import them from
there — do not redefine them:

```python
from scripts.neuron_action_head_base import (
    NeuronDenoisingWrapper,
    NeuronActionHeadBase,
    NeuronDenoisingConfig,
    ConditioningContract,
)
from scripts.cross_attention_nki import cross_attention_kernel, get_tile_size
```

**MANDATORY: DiT model construction must happen in `load_module()`, NOT `__init__()`.**

`NeuronDenoisingWrapper` subclasses `ModelWrapper`. `ModelBuilder` calls `load_module()`
after initializing `parallel_state`. If you construct the DiT model in `__init__()`,
`parallel_state` is not active, `ColumnParallelLinear` silently falls back to `nn.Linear`,
the NEFF is TP=1, weight sharding fails, and NEFF loading fails with
"Expected weight tensors for N ranks. Received 1."

```python
# CORRECT
class NeuronGrootDenoisingWrapper(NeuronDenoisingWrapper):
    def __init__(self, config):
        nn.Module.__init__(self)   # bypass ModelWrapper.__init__ — it's LLM-oriented
        self.config = config
        self.model = None          # DO NOT construct DiT here
        self._preload_sd = None

    def load_module(self):
        # parallel_state is active here — ColumnParallelLinear uses real TP
        self.model = GR00TDiTModel(self.config)
        if self._preload_sd is not None:
            self.model.load_state_dict(self._preload_sd, strict=False)
        self.model = self.model.bfloat16().eval()

    def forward(self, noisy_actions, conditioning_tokens,
                timestep_embedding, attention_mask):
        return self.model(noisy_actions, conditioning_tokens,
                          timestep_embedding, attention_mask)

# WRONG — produces silent TP=1 and broken weight sharding
class NeuronGrootDenoisingWrapper(NeuronDenoisingWrapper):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.model = GR00TDiTModel(config)  # WRONG — parallel_state not active
```

### 2.1 Compiled Graph Boundary

The following goes INSIDE the compiled graph (inside `NeuronDenoisingWrapper`):
- Single forward pass of the denoiser (one denoising step)
- Self-attention over action tokens
- Cross-attention from action tokens into VLM conditioning tokens
- Timestep embedding MLP (sinusoidal frequency computation stays on CPU,
  only the projected embedding enters the graph)
- AdaLN or FiLM conditioning if present
- Output projection to action space

The following stays OUTSIDE the compiled graph (on the CPU side):
- The N-step denoising loop
- Noise schedule computation (timestep sequence, sigma values)
- Any Python control flow over denoising steps
- VLM subgraph execution (runs once before the loop starts)

### 2.2 Interface Specification

```python
class NeuronDenoisingWrapper(ModelWrapper):
    """
    Wraps the single-step denoiser forward for NxDI compilation.
    The denoising loop is owned by NeuronActionHeadBase.generate_actions(),
    not here.
    """

    def forward(
        self,
        noisy_actions,        # [B, action_chunk_size, action_dim]        BF16
        conditioning_tokens,  # [B, num_conditioning_tokens, hidden_size] BF16
        timestep_embedding,   # [B, timestep_embed_dim]                   BF16
        attention_mask,       # [B, 1, action_chunk_size,
                              #    num_conditioning_tokens]                INT32
    ) -> torch.Tensor:        # [B, action_chunk_size, action_dim]        BF16
        ...
```

All four input shapes must be static at compile time. See Section 2.3 for how
to choose each dimension.

Note: the raw timestep scalar is NOT passed into the compiled graph. The caller
computes `timestep_embedding` on CPU (sinusoidal encoding + linear projection)
and passes the projected embedding. This avoids dynamic scalar-to-embedding
computation inside the traced graph.

### 2.3 Static Shape Contract

Three dimensions must be pinned before compilation. A mismatch on any of these
causes a runtime shape error that is not caught until the integration smoke test.

**`action_chunk_size`** — number of action steps per chunk. Read from the model
config (e.g. `chunk_size`, `pred_horizon`, `action_horizon`). Typical values:
16 for GR00T, 100 for Diffusion Policy. This is fixed for a given model and does
not need bucketing.

**`num_conditioning_tokens`** — number of VLM output tokens passed to the
denoiser as cross-attention KV. This is the critical cross-subgraph contract:
it must exactly match the sequence length produced by the VLM subgraph for the
chosen vision bucket. Derive it as:
num_conditioning_tokens = num_text_tokens + num_vision_tokens_for_bucket
If the VLM uses multiple vision buckets, compile a separate denoiser bucket for
each, or pad the VLM output to a fixed maximum and use that fixed length.

**`timestep_embed_dim`** — output dimension of the timestep embedding projection.
Read from the model config (e.g. `time_embed_dim`, `d_model`).

### 2.4 Bucketing Strategy

Unlike the VLM, the denoiser does not bucket over sequence length. Bucket only if:
- `num_conditioning_tokens` varies across calls (i.e. multiple vision buckets in
  the VLM) — create one denoiser bucket per VLM vision bucket
- `action_chunk_size` is variable (rare) — bucket over chunk size

For most VLAs (GR00T, SmolVLA) these are all fixed, so a single bucket suffices.
State this explicitly in `notes.md` so future maintainers know it was a deliberate
choice, not an oversight.

### 2.5 Conditioning Token Optimization

Because the VLM output is identical across all N denoising steps, the KV
projection for the cross-attention can be computed once before the loop and
cached. This means:

1. Run the VLM subgraph once → get `conditioning_tokens`
2. Optionally project to K, V outside the compiled graph and pass projected KV
   directly → saves N−1 redundant KV projections inside the loop
3. If KV projection is kept inside the compiled graph, the compiler will
   recompute it on every step — correct but wasteful

Flag the KV projection location as a decision point in notes.md. For latency-
priority single-environment inference (batch size 1), precomputing KV outside
the loop is recommended.

### 2.6 NKI Injection Point

The cross-attention from action tokens into conditioning tokens is the hottest
op in the action head and the primary target for a custom NKI kernel. The
favorable properties for NKI optimization are:
- Fixed K/V shapes (conditioning tokens are static per call)
- Small Q sequence (action_chunk_size is typically 16-100, well within SBUF)
- BF16 throughout

To inject an NKI kernel for this attention:
1. Identify the cross-attention module in the translated block
2. Replace the `F.scaled_dot_product_attention` call with an NKI kernel call
   using `nki.language.nc` tile primitives
3. Re-run `test_block_correctness` after injection to verify numerical equivalence
   is maintained within `atol=1e-3`

Do not write the NKI kernel until after the full port compiles and passes the
definition of done in Part 5. Profile first with Neuron Explorer to confirm
cross-attention is actually the bottleneck before investing in the kernel.

---

## Part 3: The CPU-Side Denoising Loop

The denoising loop lives in `NeuronActionHeadBase.generate_actions()` and calls
the compiled `NeuronDenoisingWrapper` N times. It must not be inside any traced
module.

```python
def generate_actions(
    self,
    conditioning_tokens,  # [B, num_conditioning_tokens, hidden_size] — VLM output
    num_steps: int,       # Python int, NOT a tensor
) -> torch.Tensor:        # [B, action_chunk_size, action_dim]

    # 1. Compute noise schedule on CPU
    timesteps = self._get_timestep_sequence(num_steps)  # list of Python floats

    # 2. Sample initial noise on CPU
    noisy_actions = torch.randn(
        B, self.action_chunk_size, self.action_dim, dtype=torch.bfloat16
    )

    # 3. Denoising loop — calls compiled graph N times
    for t in timesteps:
        timestep_emb = self._embed_timestep(t)  # CPU, returns [B, embed_dim]
        noisy_actions = self.denoising_wrapper(
            noisy_actions,
            conditioning_tokens,
            timestep_emb,
            self.attention_mask,
        )

    return noisy_actions
```

Key rules:
- `num_steps` must be a Python int. Passing it as a tensor will cause a trace
  failure if it ever enters a compiled module.
- `timesteps` is computed entirely on CPU. Only the projected `timestep_emb`
  crosses into the compiled graph.
- `conditioning_tokens` is computed once before the loop (VLM subgraph output)
  and passed unchanged on every iteration.
- For flow matching, `timesteps` is a linear sequence from 1.0 to 0.0. For
  DDPM-style, use the model's noise schedule directly.

---

## Part 4: Primitive Replacement Map

| Source op | NxDI replacement | Notes |
|---|---|---|
| Self-attention (action tokens) | `NeuronAttentionBase` subclass | Standard, same as VLM backbone |
| Cross-attention (action → conditioning) | `NeuronAttentionBase` subclass | Fixed KV shapes — flag NKI injection point |
| `nn.Linear` gate/up | `ColumnParallelLinear` | |
| `nn.Linear` down/out | `RowParallelLinear` | |
| Timestep sinusoidal encoding | Keep on CPU | Do not trace this |
| Timestep projection MLP | `ColumnParallelLinear` + `RowParallelLinear` | Input is projected embedding, not raw timestep |
| AdaLN / FiLM conditioning | Custom implementation required | No NxDI primitive — flag as deviation in notes.md |
| `nn.Embedding` (if action tokenizer present) | `ParallelEmbedding` | Only for discrete-token heads, should not appear here |
| Conv layers in action tokenizer | Unfold + `ColumnParallelLinear` | If present — rare in VLAs |

---

## Part 4b: Compiler Args for DiT Subgraphs

DiT subgraphs (denoising models, flow matching action heads) must use `-O1` only. Never use `--model-type=transformer` on a DiT.

`--model-type=transformer` replaces the native softmax with a custom NKI kernel that is numerically inaccurate for DiT/FLUX-style attention. Verified effect: cosine similarity drops to 0.916, 37% mean error per step, unusable output.

The correct default is set in `NeuronActionHeadBase.get_compiler_args()`:
```python
def get_compiler_args(self) -> str:
    return (
        "--auto-cast=none "
        "-O1 "
        "--tensorizer-options='"
        "--enable-ccop-compute-overlap "
        "--cc-pipeline-tiling-factor=1'"
    )
```

If a subclass overrides `get_compiler_args()`, verify the override does not re-add `--model-type=transformer`.

---

## Part 5: Gotchas

**Dynamic iteration count** — `num_steps` must never enter the compiled graph as
a tensor. Intercept at `generate_actions()` and convert to Python int before the loop.

**Dynamic constants in denoising forward** — `torch.arange` for position IDs,
`torch.ones` for attention masks, RoPE frequency computation — pre-compute ALL of
these in `__init__` as `register_buffer()`. With 32 DiT layers this causes
`[Errno 36] File name too long` at compile time. Fix: move every dynamic constant
out of `forward()` into `__init__` as `register_buffer()`.

**TP=1 from wrong construction order** — if the DiT model is constructed in
`__init__()` instead of `load_module()`, the NEFF is silently TP=1, weight
sharding produces 8 identical copies instead of 8 shards, and NEFF loading
fails with "Expected weight tensors for N ranks. Received 1." See the correct
load_module pattern in Part 2.

**`load_state_dict` must accept `**kwargs`** — override signature:
`(self, state_dict, strict=True, **kwargs)`. `torch_neuronx` passes `assign=True`
internally. Not forwarding causes a TypeError that, if swallowed, produces a
silent missing-NEFF failure.

**Never wrap `compile_denoiser()` in a bare exception handler** — failures
must propagate. A swallowed exception produces a missing NEFF with no error.

**Noise schedule on CPU vs device** — the full schedule array (all timesteps,
all sigma values) must be computed and stored on CPU before the loop. Only the
scalar or projected embedding for the current step crosses into the compiled
graph. Passing a schedule tensor into the graph will either fail tracing or cause
unexpected dtype casting to FP32.

**Shape contract violations** — the most common runtime failure. If the VLM
subgraph pads its output to a different length than `num_conditioning_tokens` in
the denoiser bucket, the cross-attention will fail with a shape error. Catch this
in the integration smoke test by asserting
`conditioning_tokens.shape[1] == self.num_conditioning_tokens` before the loop.

**Conditioning token shape pinning** — if the VLM produces variable-length output
across vision buckets, the denoiser must receive a padded and masked version at a
fixed length. The attention mask must be passed as a compiled graph input (not
hardcoded inside the wrapper) so that padding positions are correctly zeroed out
in the cross-attention scores.

**AdaLN inside the compiled graph** — adaptive layer norm requires element-wise
scaling by a timestep-derived vector, which is computed from `timestep_emb` inside
the denoiser forward. This is traceable as long as the scale/shift vectors are
derived from the input tensor (not from Python scalars or control flow). If the
original model uses Python conditionals to switch AdaLN behavior based on the
timestep value, those conditionals must be removed and replaced with tensor ops.

**BF16 accumulation across steps** — small numerical errors in BF16 can compound
across N denoising steps. The definition of done requires a per-step MSE plot
against the HF reference to confirm errors stay flat. If MSE grows monotonically,
the likely cause is a dtype cast somewhere inside the compiled graph converting
BF16 to FP32 and back.

---

## Part 6: Definition of Done

All six criteria must pass before the action head port is considered complete.
Do not proceed to packaging until all pass.

1. CPU unit test for the cross-attention block passes `test_block_correctness`
   with synced weights within `atol=1e-3`
2. CPU unit test for the DiT MLP / AdaLN block passes `test_block_correctness`
   within `atol=1e-3`
3. Full N-step denoising loop on CPU matches HF reference action chunk output
   within `atol=1e-2` (looser tolerance acceptable due to BF16 accumulation
   across steps)
4. Denoiser subgraph compiles to NEFF with zero CPU fallback ops — verify by
   running `torch_neuronx.analyze` on the wrapper before compilation and
   confirming all ops are Neuron-supported
5. Integration smoke test passes — VLM output fed into action head, full
   denoising loop completes, output action chunk is non-degenerate (no NaN,
   no all-zeros, values within physically plausible action range for the robot)
6. Per-step MSE between Neuron and HF CPU reference stays flat across all N
   denoising steps — plot MSE at each step and confirm no monotonic growth,
   indicating no error accumulation in the loop