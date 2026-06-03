import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        assert d_k % 2 == 0, f"{d_k} % 2 != 0"
        super().__init__()

        # Frequencies for each dimension pair: θ_j = theta^(-2j/d_k), j ∈ [0, d_k/2)
        j: torch.Tensor = torch.arange(
            start=0, end=d_k // 2, device=device, dtype=torch.float32
        )
        freqs: torch.Tensor = theta ** (-2 * j / d_k)

        # Precompute cos(pos * θ_j) and sin(pos * θ_j) for all positions
        positions = torch.arange(end=max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, freqs)  # (max_seq_len, d_k/2)

        self.register_buffer("cos", angles.cos(), persistent=True)
        self.register_buffer("sin", angles.sin(), persistent=True)

    def forward(
        self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions: torch.Tensor
    ) -> torch.Tensor:
        # Lookup cos/sin for the given token positions
        cos = self.cos[token_positions]  # (seq_len, d_k/2)
        sin = self.sin[token_positions]  # (seq_len, d_k/2)

        # Reshape x into dimension pairs: (..., seq_len, d_k/2, 2)
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        x0, x1 = x_pairs[..., 0], x_pairs[..., 1]

        # Apply rotary rotation to each pair
        x0_new = x0 * cos - x1 * sin
        x1_new = x1 * cos + x0 * sin

        return torch.stack([x0_new, x1_new], dim=-1).view_as(x)


if __name__ == "__main__":
    # === Demo 1: single vector at position 0 vs position 1 ===
    # Shows how RoPE rotates a vector differently depending on position
    print("=" * 60)
    print("Demo 1: same vector at pos 0 vs pos 1 — rotation changes")
    print("=" * 60)

    d_k = 4  # small dim so we can read the numbers
    rope = RotaryPositionEmbedding(theta=2.0, d_k=d_k, max_seq_len=4)

    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # (1, d_k)
    out_0 = rope(x, torch.tensor([0]))  # position 0
    out_1 = rope(x, torch.tensor([1]))  # position 1

    print(f"input:         {x.flatten().tolist()}")
    print(f"position 0:    {out_0.flatten().tolist()}")
    print(f"position 1:    {out_1.flatten().tolist()}")
    print()

    # === Demo 2: visualize rotation on a 2D plane ===
    # With d_k=2, each token is just (x₀, x₁) — a single 2D point.
    # RoPE rotates it by angle = pos * θ₀
    print("=" * 60)
    print("Demo 2: 2D rotation — angle = pos * theta^(-0/2) = pos * 1.0")
    print("=" * 60)

    d_k = 2
    rope2 = RotaryPositionEmbedding(theta=2.0, d_k=d_k, max_seq_len=4)
    # θ₀ = 2^(-0/2) = 1.0 → rotation per step = 1 radian ≈ 57.3°
    # At pos 0: rotate by 0 rad
    # At pos 1: rotate by 1 rad
    # At pos 2: rotate by 2 rad

    x2 = torch.tensor([[1.0, 0.0]])  # unit vector along x₀

    import math

    for pos in range(4):
        out = rope2(x2, torch.tensor([pos]))
        angle = pos * 1.0  # θ₀ = 1.0
        print(
            f"pos {pos}: x0'={out[0,0]:.4f}, x1'={out[0,1]:.4f}  "
            f"(cos({angle:.2f})={math.cos(angle):.4f}, sin({angle:.2f})={math.sin(angle):.4f})"
        )

    # === Demo 3: compare two positions with the same input ===
    # Shows that RoPE gives different results at different positions
    # even when the input vector is identical
    print()
    print("=" * 60)
    print("Demo 3: identical tokens at different positions → different outputs")
    print("=" * 60)

    d_k = 4
    rope3 = RotaryPositionEmbedding(theta=2.0, d_k=4, max_seq_len=4)
    x_batch = torch.stack([torch.tensor([1.0, 0.0, 0.0, 0.0])] * 3)  # (3, d_k)
    positions = torch.tensor([0, 1, 2])
    out_batch = rope3(x_batch, positions)

    for i, pos in enumerate(positions.tolist()):
        print(f"input same for all, pos {pos}: {out_batch[i].tolist()}")

    # === Demo 4: apply to a (batch, seq_len, d_k) tensor ===
    # This is the real usage pattern in a transformer
    print()
    print("=" * 60)
    print("Demo 4: batched (batch=2, seq_len=3, d_k=4) — real usage")
    print("=" * 60)

    d_k = 4
    rope4 = RotaryPositionEmbedding(theta=2.0, d_k=d_k, max_seq_len=8)
    x4 = torch.randn(2, 3, d_k)
    positions4 = torch.arange(3)  # pos [0, 1, 2] for all batches
    out4 = rope4(x4, positions4)
    print(f"input shape:  {x4.shape}")
    print(f"output shape: {out4.shape}")
    print(f"input[0, 0]:  {x4[0, 0].tolist()}")
    print(f"output[0, 0]: {out4[0, 0].tolist()}")
