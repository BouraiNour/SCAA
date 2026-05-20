"""
Anatomical encoder with organ‑map injection at each down‑sampling stage.

The encoder reduces spatial resolution by a factor of 16 (256×256 → 16×16)
over four stages, while channel depth grows 3 → 128 → 256 → 512 → 256 (latent).
At each stage, the organ maps from the attention module are resized and added
as a residual signal, providing anatomical guidance throughout the hierarchy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualBlock, AttentionBlock


class AnatomicalEncoder(nn.Module):
    """4‑stage convolutional encoder with anatomical conditioning.

    Each stage consists of residual blocks and a strided convolution for
    downsampling. Before each stage, organ maps are injected via a 1×1
    convolution and added to the feature maps, allowing the encoder to
    use anatomical priors at multiple scales.

    Args:
        in_channels: Number of input image channels (default 3).
        latent_dim: Dimension of the output latent tensor.
        num_organs: Number of organ masks to expect from the attention module.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 256,
                 num_organs: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        # Initial projection
        self.stem = nn.Conv2d(in_channels, 128, 3, padding=1)

        # Organ map projections (one per stage)
        self.organ_proj1 = nn.Conv2d(num_organs, 128, 1)
        self.organ_proj2 = nn.Conv2d(num_organs, 256, 1)
        self.organ_proj3 = nn.Conv2d(num_organs, 512, 1)
        self.organ_proj4 = nn.Conv2d(num_organs, 512, 1)

        # Stage 1: 256×256 → 128×128, 128 → 256 channels
        self.down1 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
        )
        # Stage 2: 128×128 → 64×64, 256 → 512 channels
        self.down2 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
        )
        # Stage 3: 64×64 → 32×32, 512 channels (with attention bottleneck)
        self.down3 = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            AttentionBlock(512),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
        )
        # Stage 4: 32×32 → 16×16, 512 channels (with attention bottleneck)
        self.down4 = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            AttentionBlock(512),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            AttentionBlock(512),
            ResidualBlock(512),
        )

        # Project to latent dimension
        self.to_latent = nn.Conv2d(512, latent_dim, 1)

    def _inject_organ(self, x: torch.Tensor, organ_maps: torch.Tensor,
                      proj: nn.Conv2d) -> torch.Tensor:
        """Resize organ maps to match x and add as a residual."""
        om = F.interpolate(organ_maps, x.shape[-2:],
                           mode='bilinear', align_corners=False)
        return x + proj(om)

    def forward(self, x: torch.Tensor,
                organ_maps: torch.Tensor) -> torch.Tensor:
        """Encode an image with anatomical guidance.

        Args:
            x: Input image tensor (B, C, H, W).
            organ_maps: Organ masks from anatomical attention (B, K, H, W).

        Returns:
            Latent tensor (B, latent_dim, H/16, W/16).
        """
        h = self.stem(x)
        h = self._inject_organ(h, organ_maps, self.organ_proj1)

        h = self.down1(h)
        h = self._inject_organ(h, organ_maps, self.organ_proj2)

        h = self.down2(h)
        h = self._inject_organ(h, organ_maps, self.organ_proj3)

        h = self.down3(h)
        h = self._inject_organ(h, organ_maps, self.organ_proj4)

        h = self.down4(h)
        h = self.bottleneck(h)
        h = self.to_latent(h)          # (B, latent_dim, H/16, W/16)
        return h
