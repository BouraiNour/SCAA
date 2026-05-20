"""
Anatomical decoder with organ‑map modulation.

The decoder mirrors the encoder: four upsampling stages (16× upscaling)
with organ‑map modulation applied after each stage. Modulation is implemented
as a learned spatial scale and shift conditioned on the organ maps, allowing
the decoder to use anatomical information during reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualBlock, AttentionBlock


class OrgMapModulation(nn.Module):
    """Spatially‑adaptive feature modulation conditioned on organ maps.

    Given organ maps, learns a per‑channel scale (gamma) and shift (beta)
    via 1×1 convolutions. The modulation is applied as:

        output = x * (1 + gamma) + beta

    Weights are initialised to zero, so the module starts as an identity.

    Args:
        num_organs: Number of organ channels in the conditioning maps.
        channels: Number of feature channels to modulate.
    """

    def __init__(self, num_organs: int, channels: int):
        super().__init__()
        self.gamma_conv = nn.Conv2d(num_organs, channels, kernel_size=1)
        self.beta_conv  = nn.Conv2d(num_organs, channels, kernel_size=1)

        # Zero initialisation → identity at start of training
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.zeros_(self.beta_conv.weight)
        if self.gamma_conv.bias is not None:
            nn.init.zeros_(self.gamma_conv.bias)
        if self.beta_conv.bias is not None:
            nn.init.zeros_(self.beta_conv.bias)

    def forward(self, x: torch.Tensor,
                organ_maps: torch.Tensor) -> torch.Tensor:
        """Apply modulation.

        Args:
            x: Feature map (B, C, H, W).
            organ_maps: Organ masks from anatomical attention (B, K, H_map, W_map).

        Returns:
            Modulated feature map (B, C, H, W).
        """
        maps = F.interpolate(organ_maps, x.shape[2:],
                             mode='bilinear', align_corners=False)
        gamma = self.gamma_conv(maps)
        beta  = self.beta_conv(maps)
        return x * (1.0 + gamma) + beta


class AnatomicalDecoder(nn.Module):
    """4‑stage decoder with organ‑map modulation at each upsampling stage.

    Symmetric to the encoder: the latent is projected back to 512 channels,
    passed through a bottleneck, then upsampled through four stages, each
    consisting of a transposed convolution, residual blocks, and attention.
    After each stage (except the last), an OrgMapModulation block injects
    anatomical information.

    Args:
        latent_dim: Dimension of the input latent tensor.
        out_channels: Number of output image channels (default 3).
        num_organs: Number of organ mask channels expected.
    """

    def __init__(self, latent_dim: int = 256, out_channels: int = 3,
                 num_organs: int = 3):
        super().__init__()

        self.from_latent = nn.Conv2d(latent_dim, 512, 1)

        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            AttentionBlock(512),
            ResidualBlock(512),
        )

        # Stage 1: 16×16 → 32×32, 512 channels
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            ResidualBlock(512),
            AttentionBlock(512),
            ResidualBlock(512),
        )
        # Stage 2: 32×32 → 64×64, 512 → 256 channels
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            ResidualBlock(256),
            AttentionBlock(256),
            ResidualBlock(256),
        )
        # Stage 3: 64×64 → 128×128, 256 → 128 channels
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        # Stage 4: 128×128 → 256×256, 128 channels
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # Modulation blocks (applied after up1, up2, up3)
        self.mod1 = OrgMapModulation(num_organs, 512)
        self.mod2 = OrgMapModulation(num_organs, 256)
        self.mod3 = OrgMapModulation(num_organs, 128)

        # Output head
        self.head = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor,
                organ_maps: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor into an image.

        Args:
            z: Latent tensor (B, latent_dim, H/16, W/16).
            organ_maps: Organ masks from anatomical attention (B, K, H, W).

        Returns:
            Reconstructed image (B, out_channels, H, W), values in [0, 1].
        """
        x = self.from_latent(z)
        x = self.bottleneck(x)

        x = self.up1(x)
        x = self.mod1(x, organ_maps)

        x = self.up2(x)
        x = self.mod2(x, organ_maps)

        x = self.up3(x)
        x = self.mod3(x, organ_maps)

        x = self.up4(x)    # no modulation before the output head
        x = self.head(x)
        return x
