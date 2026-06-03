import torch
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[torch.Tensor, "... vocab_size"],
    targets: Int[torch.Tensor, " ..."],
) -> Float[torch.Tensor, ""]:
    # ℓ = -log(softmax(o)[target])
    #   = -(o[target] - log(Σ_j exp(o[j])))
    #   = log(Σ_j exp(o[j])) - o[target]
    #     ^^^^^^^^^^^^^^^^^ = logsumexp(o)
    #
    # Numerically stable logsumexp: subtract max before exp
    #   logsumexp(o) = max(o) + log(Σ_j exp(o[j] - max(o)))
    max_val = inputs.max(dim=-1, keepdim=True).values
    logsumexp = (inputs - max_val).exp().sum(dim=-1).log() + max_val.squeeze(-1)

    # o[target]: gather the logit at the target class index
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Per-example loss averaged over the batch
    return (logsumexp - target_logits).mean()
