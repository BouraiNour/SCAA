"""
Fundamental neural network building blocks.

Classes:
    ResidualBlock: Standard residual block with GroupNorm and SiLU.
    AttentionBlock: Efficient single‑head spatial self‑attention (O(H*W)² complexity
                    but applied only to small feature maps in the bottleneck).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard residual block with GroupNorm and SiLU activation.

    Uses two 3×3 convolutions with a skip connection. GroupNorm is applied
    before each activation. The number of groups is automatically adjusted
    to be a divisor of the channel count.

    Args:
        channels: Number of input/output channels.
        groups: Desired number of groups for GroupNorm (will be reduced if
                channels is not divisible).
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        # Ensure groups divides channels
        groups = min(groups, channels)
        while channels % groups != 0:
            groups -= 1

        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of same shape.
        """
        return x + self.block(x)


class AttentionBlock(nn.Module):
    """Efficient single‑head spatial self‑attention.

    Projects input into queries, keys, values via 1×1 convolutions,
    computes scaled dot‑product attention over spatial locations,
    then projects back. Designed for small spatial feature maps
    (e.g., 16×16 or 32×32) where the O(HW²) complexity is acceptable.

    Args:
        channels: Number of input channels.
        groups: Number of groups for the pre‑norm layer (GroupNorm).
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        groups = min(groups, channels)
        while channels % groups != 0:
            groups -= 1

        self.norm = nn.GroupNorm(groups, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)   # combined Q/K/V
        self.proj = nn.Conv2d(channels, channels, 1)       # output projection
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of same shape, added to the residual.
        """
        B, C, H, W = x.shape
        h = self.norm(x)

        # Compute queries, keys, values
        qkv = self.qkv(h).reshape(B, 3, C, H * W)   # [B, 3, C, HW]
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]   # each [B, C, HW]

        # Attention scores
        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale   # [B, HW, HW]
        attn = attn.softmax(dim=-1)

        # Weighted sum of values
        out = torch.bmm(v, attn.permute(0, 2, 1))              # [B, C, HW]
        out = self.proj(out.reshape(B, C, H, W))

        return x + out
