import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.weights = nn.Parameter(
            torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        )
        self._reset_parameter()

    def _reset_parameter(self):
        nn.init.trunc_normal_(tensor=self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights[x]
