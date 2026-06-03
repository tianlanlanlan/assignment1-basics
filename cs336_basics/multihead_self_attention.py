import math

import torch
from torch import nn

from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionEmbedding | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model={d_model} must be divisible by num_heads={num_heads}"
        self.d_model = d_model
        self.n_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # All projections: (B, T, d_model) -> (B, T, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads: (B, T, d_model) -> (B, T, H, d_k) -> (B, H, T, d_k)
        Q = torch.einsum("bthn->bhtn", Q.view(B, T, self.n_heads, self.d_k))
        K = torch.einsum("bthn->bhtn", K.view(B, T, self.n_heads, self.d_k))
        V = torch.einsum("bthn->bhtn", V.view(B, T, self.n_heads, self.d_k))

        # Apply RoPE to Q and K if configured
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(T, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        out = scaled_dot_product_attention(Q, K, V, mask)  # (B, H, T, d_k)

        # Merge heads: (B, H, T, d_k) -> (B, T, d_model)
        out = torch.einsum("bhtn->bthn", out).reshape(B, T, self.d_model)

        return self.o_proj(out)
