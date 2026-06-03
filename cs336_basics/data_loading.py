import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start, size=batch_size)

    xs = np.stack([dataset[i : i + context_length] for i in start_indices])
    ys = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_indices])

    return torch.tensor(xs, device=device, dtype=torch.long), torch.tensor(ys, device=device, dtype=torch.long)
