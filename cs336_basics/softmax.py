import torch
from jaxtyping import Float


def softmax(x: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = (x - x_max).exp()
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


if __name__ == "__main__":
    x = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])  # shape (2, 3)

    x_max = x.max(dim=1, keepdim=True)

    print(f"x_max: {x_max}")
