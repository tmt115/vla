"""
neuron_action_head_base.py

Abstract base class for porting VLA action heads to AWS Trainium via NxDI.
Subclass this for each model-specific action head implementation.

Model-specific subclasses (written by the port agent, not here):
    NeuronGrootActionHead(NeuronActionHeadBase)
    NeuronSmolVLAActionHead(NeuronActionHeadBase)

Usage in port agent (Phase 2):
    - Subclass NeuronActionHeadBase
    - Implement all abstract methods
    - Implement NeuronDenoisingWrapper.forward() for the single compiled step
    - The denoising loop lives in generate_actions() here — do not move it
      into the compiled wrapper

NOTE: Verify all import paths against the installed NxDI version on your
Trainium instance before running. Paths below match NxDI venv layout as of
neuronx_distributed_inference 2.28.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Verify these paths on the Trainium instance:
#   python -c "from neuronx_distributed_inference.models.model_base import
#              NeuronApplicationBase; print('ok')"
try:
    from neuronx_distributed_inference.models.model_base import NeuronApplicationBase
    from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
    from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    _NEURON_AVAILABLE = True
except ImportError:
    # CPU fallback — allows unit testing without Neuron hardware
    _NEURON_AVAILABLE = False
    NeuronApplicationBase = object
    ModelWrapper = nn.Module
    InferenceConfig = object
    NeuronConfig = object


# ---------------------------------------------------------------------------
# Conditioning contract dataclass
# ---------------------------------------------------------------------------

class ConditioningContract:
    """
    Defines the static shape agreement between the VLM subgraph and the
    action head subgraph. Must be identical on both sides at compile time.

    Attributes:
        num_conditioning_tokens: Number of VLM output tokens passed to the
            denoiser as cross-attention KV. Derived as:
                num_text_tokens + num_vision_tokens_for_bucket
            Must match the VLM subgraph output sequence length exactly.
        conditioning_hidden_size: Hidden dimension of conditioning tokens.
            Must match VLM output hidden size.
        dtype: Expected dtype of conditioning tokens (typically torch.bfloat16).
    """

    def __init__(
        self,
        num_conditioning_tokens: int,
        conditioning_hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_conditioning_tokens = num_conditioning_tokens
        self.conditioning_hidden_size = conditioning_hidden_size
        self.dtype = dtype

    def validate(self, conditioning_tokens: torch.Tensor) -> None:
        """
        Assert that a conditioning tensor matches this contract.
        Call this at the start of generate_actions() to catch shape
        mismatches before they produce cryptic errors inside the loop.
        """
        assert conditioning_tokens.shape[1] == self.num_conditioning_tokens, (
            f"Conditioning token count mismatch: expected "
            f"{self.num_conditioning_tokens}, got {conditioning_tokens.shape[1]}. "
            f"Check that the VLM vision bucket matches the denoiser bucket."
        )
        assert conditioning_tokens.shape[2] == self.conditioning_hidden_size, (
            f"Conditioning hidden size mismatch: expected "
            f"{self.conditioning_hidden_size}, got {conditioning_tokens.shape[2]}."
        )
        assert conditioning_tokens.dtype == self.dtype, (
            f"Conditioning dtype mismatch: expected {self.dtype}, "
            f"got {conditioning_tokens.dtype}."
        )


# ---------------------------------------------------------------------------
# NeuronDenoisingWrapper — base for the compiled single-step forward
# ---------------------------------------------------------------------------

class NeuronDenoisingWrapper(ModelWrapper):
    """
    Base wrapper for the single-step denoiser forward. This is what gets
    compiled to a NEFF. The N-step denoising loop lives in
    NeuronActionHeadBase.generate_actions(), NOT here.

    Subclass this and implement forward() for each model.

    Compiled graph inputs (all shapes must be STATIC at compile time):
        noisy_actions:        [B, action_chunk_size, action_dim]         BF16
        conditioning_tokens:  [B, num_conditioning_tokens, hidden_size]  BF16
        timestep_embedding:   [B, timestep_embed_dim]                    BF16
        attention_mask:       [B, 1, action_chunk_size,
                               num_conditioning_tokens]                  INT32

    Compiled graph output:
        denoised_actions:     [B, action_chunk_size, action_dim]         BF16

    IMPORTANT: Do NOT pass raw timestep scalars or noise schedule arrays into
    this wrapper. The caller computes timestep_embedding on CPU (sinusoidal
    encoding + linear projection) and passes only the projected embedding.
    This avoids dynamic scalar-to-embedding computation inside the traced graph.

    IMPORTANT: Do NOT include the denoising loop here. Any Python-level
    iteration over timesteps must live in generate_actions().
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(
        self,
        noisy_actions: torch.Tensor,
        conditioning_tokens: torch.Tensor,
        timestep_embedding: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single denoising step forward pass. Implement in model-specific subclass.

        Args:
            noisy_actions:       [B, action_chunk_size, action_dim]        BF16
            conditioning_tokens: [B, num_conditioning_tokens, hidden_size] BF16
            timestep_embedding:  [B, timestep_embed_dim]                   BF16
            attention_mask:      [B, 1, action_chunk_size,
                                  num_conditioning_tokens]                  INT32

        Returns:
            denoised_actions:    [B, action_chunk_size, action_dim]        BF16
        """
        raise NotImplementedError

    def input_generator(self) -> List[Tuple[torch.Tensor, ...]]:
        """
        Returns example inputs for XLA tracing. Override if the default
        zero-tensor inputs are insufficient for your model.
        Shapes must exactly match forward() signature.
        """
        cfg = self.config
        B = cfg.neuron_config.batch_size
        return [(
            torch.zeros(B, cfg.action_chunk_size, cfg.action_dim,
                        dtype=torch.bfloat16),
            torch.zeros(B, cfg.num_conditioning_tokens,
                        cfg.conditioning_hidden_size, dtype=torch.bfloat16),
            torch.zeros(B, cfg.timestep_embed_dim, dtype=torch.bfloat16),
            torch.zeros(B, 1, cfg.action_chunk_size,
                        cfg.num_conditioning_tokens, dtype=torch.int32),
        )]


# ---------------------------------------------------------------------------
# NeuronActionHeadBase — owns the lifecycle and denoising loop
# ---------------------------------------------------------------------------

class NeuronActionHeadBase(NeuronApplicationBase):
    """
    Abstract base class for VLA action heads on Trainium.

    Owns:
        - compile_denoiser(): compiles the single-step forward to a NEFF
        - generate_actions(): runs the N-step denoising loop on CPU,
          calling the compiled wrapper each step
        - get_conditioning_contract(): returns the ConditioningContract
          that the VLM subgraph must satisfy

    Does NOT own:
        - The VLM subgraph (separate compiled graph)
        - The noise schedule (computed on CPU in generate_actions)
        - Weight loading for the VLM backbone

    Subclass this and implement all abstract methods for each model.
    See NeuronGrootActionHead and NeuronSmolVLAActionHead for examples.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.denoising_wrapper: Optional[NeuronDenoisingWrapper] = None

    # ------------------------------------------------------------------
    # Abstract methods — implement in model-specific subclass
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_conditioning_contract(self) -> ConditioningContract:
        """
        Return the ConditioningContract this action head expects from the
        VLM subgraph. Called during integration to verify shape agreement.

        Example:
            return ConditioningContract(
                num_conditioning_tokens=512,
                conditioning_hidden_size=2048,
                dtype=torch.bfloat16,
            )
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_denoising_wrapper(self) -> NeuronDenoisingWrapper:
        """
        Instantiate and return the model-specific NeuronDenoisingWrapper.
        Called by compile_denoiser().
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_timestep_sequence(self, num_steps: int) -> List[float]:
        """
        Return the ordered list of timestep values for the denoising loop.
        All values must be Python floats — never tensors.

        For flow matching (GR00T, SmolVLA, pi0):
            return [1.0 - i / num_steps for i in range(num_steps)]

        For DDPM-style:
            return list of beta schedule values in reverse order
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _embed_timestep(self, t: float) -> torch.Tensor:
        """
        Compute the timestep embedding for a single timestep value on CPU.
        Returns [B, timestep_embed_dim] BF16 tensor.

        t must be a Python float, NOT a tensor. Sinusoidal frequency
        computation stays here on CPU — never inside the compiled wrapper.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_attention_mask(self) -> torch.Tensor:
        """
        Build the static cross-attention mask for the denoising step.
        Returns [B, 1, action_chunk_size, num_conditioning_tokens] INT32.

        Called once during compile_denoiser() and stored as a buffer.
        If conditioning tokens are padded to a fixed length, zeros out
        padding positions here.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete methods — do not override unless necessary
    # ------------------------------------------------------------------

    def compile_denoiser(self, save_path: str) -> None:
        """
        Build and compile the denoising wrapper to a NEFF, saved at save_path.

        Args:
            save_path: Directory path to save compiled artifacts.
                       e.g. "~/mymodel_port/compiled/action_head/"
        """
        self.denoising_wrapper = self._build_denoising_wrapper()
        self.attention_mask = self._build_attention_mask()
        # Compilation delegated to NxDI compile infrastructure.
        # Subclass may override to pass custom compiler args.
        self.denoising_wrapper.compile(save_path)

    def generate_actions(
        self,
        conditioning_tokens: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """
        Run the full N-step denoising loop and return the predicted action chunk.

        Args:
            conditioning_tokens: VLM output. [B, num_conditioning_tokens,
                                 hidden_size] BF16. Must satisfy
                                 get_conditioning_contract().
            num_steps: Number of denoising steps. Must be a Python int —
                       never a tensor. Dynamic step counts are not supported
                       inside compiled graphs.

        Returns:
            action_chunk: [B, action_chunk_size, action_dim] BF16
        """
        assert self.denoising_wrapper is not None, (
            "compile_denoiser() must be called before generate_actions()"
        )
        assert isinstance(num_steps, int), (
            f"num_steps must be a Python int, got {type(num_steps)}. "
            f"Dynamic step counts cannot enter the compiled graph."
        )

        # Validate conditioning tensor matches the contract
        self.get_conditioning_contract().validate(conditioning_tokens)

        B = conditioning_tokens.shape[0]
        cfg = self.config

        # 1. Sample initial noise on CPU
        noisy_actions = torch.randn(
            B, cfg.action_chunk_size, cfg.action_dim,
            dtype=torch.bfloat16,
        )

        # 2. Compute timestep sequence on CPU — all Python floats
        timesteps = self._get_timestep_sequence(num_steps)

        # 3. Denoising loop — calls compiled graph N times
        #    conditioning_tokens is identical every step (VLM output is static)
        for t in timesteps:
            timestep_emb = self._embed_timestep(t)  # [B, timestep_embed_dim] on CPU
            noisy_actions = self.denoising_wrapper(
                noisy_actions,
                conditioning_tokens,
                timestep_emb,
                self.attention_mask,
            )

        return noisy_actions

    def verify_contract_against_vlm(
        self,
        vlm_output: torch.Tensor,
    ) -> None:
        """
        Convenience method to verify VLM output satisfies this action head's
        conditioning contract before running the denoising loop.
        Call this in the integration smoke test.
        """
        self.get_conditioning_contract().validate(vlm_output)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, torch.Tensor],
        config: InferenceConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Placeholder weight conversion. Override in model-specific subclass
        to implement key renaming, weight fusion, and metadata tensor injection.
        Implemented in Phase 4 of the port.
        """
        return state_dict  # placeholder — implemented in Phase 4