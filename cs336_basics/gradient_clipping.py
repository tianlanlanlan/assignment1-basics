from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.Tensor], max_l2_norm: float) -> None:
    # Compute total L2 norm of all gradients combined
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)
