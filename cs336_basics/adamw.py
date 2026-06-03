import torch
from torch import optim


class AdamW(optim.Optimizer):
    """
    算法（每步）：
        1. 更新一阶矩 m = β₁·m + (1-β₁)·g
        2. 更新二阶矩 v = β₂·v + (1-β₂)·g²
        3. 偏差校正 m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ)
        4. Adam 更新: θ -= α·m̂/(√v̂ + ε)
        5. 解耦权重衰减: θ *= (1 - α·λ) — 与 L2 正则化的区别是这里在更新后才施加，不影响梯度
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # Bias-corrected moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.square(), alpha=1 - beta2)
                bias_corr1 = 1 - beta1**t
                bias_corr2 = 1 - beta2**t

                # Adam update (no weight decay here)
                denom = exp_avg_sq.sqrt().div_(bias_corr2**0.5).add_(eps)
                step_size = lr / bias_corr1
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

        return loss
