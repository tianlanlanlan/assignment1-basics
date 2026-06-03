import math

import torch
from jaxtyping import Float, Bool
from torch import Tensor

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    # (..., queries, d_k) x (..., keys, d_k) -> (..., queries, keys)
    # "... i k, ... j k -> ... i j" means: contract over k, keep i (queries) and j (keys)
    scores = torch.einsum("...ik,...jk->...ij", Q, K)
    # Scale by sqrt(d_k) to prevent softmax saturation
    d_k = K.shape[-1]
    scores = scores / math.sqrt(d_k)
    # Optional mask: set masked positions to -inf so softmax zeros them out
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    return attn_weights @ V
