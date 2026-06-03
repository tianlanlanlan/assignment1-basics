import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    progress = (it - warmup_iters) / max(1, cosine_cycle_iters - warmup_iters)
    cosine = 0.5 + 0.5 * math.cos(math.pi * min(progress, 1.0))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine
