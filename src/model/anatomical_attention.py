"""
Anatomical attention with sparsity‑controlled learnable priors.

Self‑supervised attention mechanism that learns to produce sparse, 
organ‑specific masks and uses them to gate features before applying
linear‑complexity spatial attention.

Classes:
    SparsityPriorGenerator: Produces temperature‑scaled soft organ masks.
    AnatomicalAttention: Full attention block with dual‑scale priors,
                         organ gating, and linear attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsityPriorGenerator(nn.Module):
    """Generates soft organ‑specific masks via a learnable temperature sigmoid.

    The module compresses the input spatially (AdaptiveAvgPool), extracts
    features, and produces K masks. A learnable temperature parameter controls
    the sharpness of the sigmoid: high temperature → near‑binary (sparse) masks,
    low temperature → soft, overlapping masks.

    Args:
        in_channels: Number of input channels.
        num_organs: Number of organ classes (K).
        init_temperature: Initial value for the learnable temperature.
        pool_size: Spatial size after adaptive pooling.
        feature_channels: Number of feature channels in the extractor.
    """

    def __init__(self, in_channels: int = 3, num_organs: int = 3,
                 init_temperature: float = 10.0, pool_size: int = 32,
                 feature_channels: int = 64):
        super().__init__()
        self.num_organs = num_organs
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, feature_channels, 3, padding=1),
            nn.GroupNorm(4, feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, num_organs, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(num_organs, num_organs, 3, padding=1),
            nn.GroupNorm(1, num_organs),
            nn.GELU(),
            nn.Conv2d(num_organs, num_organs, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate organ masks.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Organ masks (B, num_organs, H', W') where H', W' depend on
            upsampling. The final sigmoid uses the learnable temperature.
        """
        features = self.feature_extractor(x)
        return torch.sigmoid(self.temperature * features)


class AnatomicalAttention(nn.Module):
    """Linear‑complexity anatomical attention with sparsity‑controlled priors.

    This block:
    1. Generates multi‑scale organ masks (fine and coarse) and fuses them
       with a learnable weight.
    2. Computes a spatial gating map from the fused masks.
    3. Gating the input features to focus on anatomical regions.
    4. Applies linear attention (ELU kernel, O(N) complexity) to the gated
       features, producing a residual attention map.

    The output is `x + gamma * attended_features`, where gamma is a
    learnable scalar initialised to a small value for stable training.

    Args:
        in_channels: Number of input channels.
        num_organs: Number of organ classes.
        reduction: Channel reduction factor for queries/keys (kept ≥ 4).
        init_temperature: Initial temperature for the prior generators.
    """

    def __init__(self, in_channels: int = 3, num_organs: int = 3,
                 reduction: int = 4, init_temperature: float = 10.0):
        super().__init__()
        self.num_organs = num_organs
        self.reduction = reduction

        # Fine‑scale prior generator (full resolution, pool_size=32)
        self.prior_generator = SparsityPriorGenerator(
            in_channels, num_organs, init_temperature,
            pool_size=32, feature_channels=64,
        )

        # Coarse‑scale prior generator (half resolution, pool_size=16)
        self.prior_generator_coarse = SparsityPriorGenerator(
            in_channels, num_organs, init_temperature,
            pool_size=16, feature_channels=64,
        )

        # Learnable fusion weight between fine and coarse priors
        self.prior_fusion_weight = nn.Parameter(torch.ones(1) * 0.5)

        # Reduced dimension for keys/queries (minimum 4)
        reduced_raw = max(1, in_channels // reduction)
        self.reduced_dim = max(4, reduced_raw)

        # Linear attention components
        self.key_conv   = nn.Conv2d(in_channels, self.reduced_dim, 1)
        self.query_conv = nn.Conv2d(in_channels, self.reduced_dim, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)

        self.norm1 = nn.InstanceNorm2d(self.reduced_dim)
        self.norm2 = nn.InstanceNorm2d(self.reduced_dim)
        self.norm3 = nn.InstanceNorm2d(in_channels)

        # Organ gating network (16 → 32 → 1)
        self.organ_gate = nn.Sequential(
            nn.Conv2d(num_organs, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        # Learnable residual strength
        self.gamma   = nn.Parameter(torch.ones(1) * 0.05)
        self.epsilon = 1e-5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of anatomical attention.

        Args:
            x: Input image tensor (B, C, H, W).

        Returns:
            Tuple of:
                - attended_x: Attention‑enhanced features (B, C, H, W).
                - organ_maps: Fused organ masks (B, num_organs, H, W).
        """
        B, C, H, W = x.shape

        # Generate multi‑scale organ masks
        organ_maps_fine   = self.prior_generator(x)
        organ_maps_coarse = self.prior_generator_coarse(x)

        # Resize both to input resolution
        organ_maps_fine   = F.interpolate(organ_maps_fine,   (H, W),
                                          mode='bilinear', align_corners=False)
        organ_maps_coarse = F.interpolate(organ_maps_coarse, (H, W),
                                          mode='bilinear', align_corners=False)

        # Fuse with learnable weight
        w = torch.sigmoid(self.prior_fusion_weight)
        organ_maps = w * organ_maps_fine + (1.0 - w) * organ_maps_coarse

        # Anatomical gating
        organ_gate = self.organ_gate(organ_maps)   # (B, 1, H, W)
        x_gated    = x * organ_gate

        # Linear attention
        key   = self.norm1(self.key_conv(x_gated)).view(B, -1, H * W)
        query = self.norm2(self.query_conv(x_gated)).view(B, -1, H * W)
        value = self.norm3(self.value_conv(x)).view(B, -1, H * W)

        # ELU kernel for linear attention
        kernel_fn = lambda t: F.elu(t) + 1.0
        Q = kernel_fn(query).permute(0, 2, 1)       # (B, HW, reduced)
        K = kernel_fn(key)                           # (B, reduced, HW)
        V = value.permute(0, 2, 1)                   # (B, HW, C)

        # Efficient attention: (Q @ (K @ V)) / (Q @ sum(K))
        KV = torch.bmm(K, V)                         # (B, reduced, C)
        numerator = torch.bmm(Q, KV)                 # (B, HW, C)
        K_sum = K.sum(dim=-1, keepdim=True)          # (B, reduced, 1)
        denominator = torch.bmm(Q, K_sum)            # (B, HW, 1)
        denominator = torch.clamp(denominator, min=self.epsilon)

        weighted_value = numerator / denominator     # (B, HW, C)
        weighted_value = weighted_value.permute(0, 2, 1).view(B, C, H, W)

        # Residual connection
        output = x + self.gamma * weighted_value

        return output, organ_maps
