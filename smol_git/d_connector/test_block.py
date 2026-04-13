import sys
sys.path.insert(0, '/home/ubuntu/smol_port/tests')
import _transformers_compat  # bypass huggingface-hub version check — must be first
sys.path.insert(0, '/home/ubuntu/smol_port/d_connector')

import torch
from transformers.models.idefics3.modeling_idefics3 import Idefics3Connector
from nxdi_block import NeuronSmolVLMConnector
from block_testing_utils import test_block_correctness


class MockIdefics3Config:
    class text_config:
        hidden_size = 960
    class vision_config:
        hidden_size = 768
    scale_factor = 4


if __name__ == "__main__":
    B = 1
    sample = torch.randn(B, 1024, 768, dtype=torch.bfloat16)

    test_block_correctness(
        neuron_block_class=NeuronSmolVLMConnector,
        pytorch_block_class=Idefics3Connector,
        weight_mapping={"modality_projection.proj.weight": "proj.weight"},
        neuron_init_kwargs={
            "vision_hidden_size": 768,
            "text_hidden_size": 960,
            "scale_factor": 4,
        },
        pytorch_init_kwargs={"config": MockIdefics3Config()},
        example_inputs=[(torch.zeros(B, 1024, 768, dtype=torch.bfloat16),)],
        test_inputs=[(sample,)],
        reference_inputs=[(sample,)],
        checkpoint_name="d_connector.pt",
        batch_size=B,
        seq_len=1024,
        hidden_size=768,
        verbose=True,
    )
