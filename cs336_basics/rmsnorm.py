import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.eps = eps
        self.d_model = d_model

        self.gamma = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.to(torch.float32)

        variance = x.pow(2).mean(-1, keepdim=True)
        rms: torch.Tensor = torch.sqrt(variance + self.eps)
        result: torch.Tensor = (x / rms) * self.gamma
        return result.to(x_dtype)


def main():
    d_model = 8

    if torch.cuda.is_available():
        input_x = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda:0", dtype=torch.float32
        )
    else:
        input_x = torch.tensor(
            [[1.0, 2.0, 100.0], [3.0, 4.0, 5.0]],
        )
    rmsnorm = RMSNorm(d_model=3, eps=1e-5, device=input_x.device, dtype=input_x.dtype)
    out = rmsnorm(input_x)


if __name__ == "__main__":
    main()
