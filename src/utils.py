"""
Utility components for the compression model.

Currently contains the IdentityAttention module used in ablation studies.
"""

import torch
import torch.nn as nn


class IdentityAttention(nn.Module):
    """Attention stub that returns the input unchanged with zero organ maps.

    Used in ablation experiments to measure the contribution of the
    anatomical attention module by replacing it with an identity mapping.
    """

    def __init__(self, in_channels: int, num_organs: int):
        super().__init__()
        # Dummy parameter to keep the module structure consistent
        self.dummy = nn.Parameter(torch.zeros(1, num_organs, 1, 1),
                                  requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return input as‑is along with zero‑filled organ maps.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Tuple of (x, zero_organ_maps) where zero_organ_maps has shape
            (B, num_organs, H, W) filled with zeros.
        """
        B, C, H, W = x.shape
        return x, self.dummy.expand(B, -1, H, W).clone()
