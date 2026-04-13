import torch
import torch.nn as nn

try:
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
    HAS_NXD = True
except Exception:
    HAS_NXD = False


class NeuronSmolVLMConnector(nn.Module):
    def __init__(self, vision_hidden_size=768, text_hidden_size=960, scale_factor=4, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        proj_in = vision_hidden_size * (scale_factor ** 2)  # 768 * 16 = 12288

        if HAS_NXD and parallel_state.model_parallel_is_initialized():
            self.proj = ColumnParallelLinear(
                proj_in, text_hidden_size,
                bias=False, gather_output=True,
                dtype=torch.bfloat16,
            )
        else:
            self.proj = nn.Linear(proj_in, text_hidden_size, bias=False)

    def pixel_shuffle(self, x, scale_factor=2):
        # Exact replica of Idefics3Connector.pixel_shuffle
        bsz, seq, embed_dim = x.size()
        height = width = int(seq ** 0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor ** 2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor ** 2)), embed_dim * (scale_factor ** 2))
        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)  # [B, 64, 12288]
        image_hidden_states = self.proj(image_hidden_states)                               # [B, 64, 960]
        return image_hidden_states
