import torch
from torch import nn
import math


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # 𝑊 ∈ ℝ𝑑out×𝑑in
        self.weights = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        self.reset_parameter()

    def reset_parameter(self):
        std = math.sqrt(2 / (self.out_features + self.in_features))
        nn.init.trunc_normal_(
            tensor=self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T
