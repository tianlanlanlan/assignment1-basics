import torch
from torch import nn

from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RotaryPositionEmbedding
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device)
        rope = RotaryPositionEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
        self.attn = MultiheadSelfAttention(d_model, num_heads, rope=rope)
        self.ln2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: RMSNorm -> MHA -> residual
        x = x + self.attn(self.ln1(x))
        # Pre-norm: RMSNorm -> FFN -> residual
        x = x + self.ffn(self.ln2(x))
        return x
