"""
Medical image compression model using VQ‑VAE with anatomical attention.

The model applies anatomical attention to the input, encodes the result,
quantizes the latent with a residual vector quantizer, and decodes back to
an image. The forward pass returns a dictionary containing the reconstruction,
organ maps, quantizer outputs, and loss components.
"""

import torch
import torch.nn as nn
from .anatomical_attention import AnatomicalAttention
from .encoder import AnatomicalEncoder
from .quantizer import ResidualVectorQuantizer
from .decoder import AnatomicalDecoder


class MedicalCompressor(nn.Module):
    """VQ‑VAE for medical image compression with anatomical attention.

    Pipeline:
        1. AnatomicalAttention produces organ maps and attended features.
        2. AnatomicalEncoder encodes the attended features to a latent.
        3. ResidualVectorQuantizer quantizes the latent.
        4. AnatomicalDecoder reconstructs the image from quantized latent
           and organ maps.

    Args:
        in_channels: Number of input image channels (default 3).
        num_organs: Number of organ classes for the attention module.
        latent_dim: Dimensionality of the latent space.
        init_temperature: Initial temperature for sparsity priors.
        n_codebooks: Number of residual VQ codebooks.
        codebook_size: Number of entries per codebook.
    """

    def __init__(self, in_channels: int = 3, num_organs: int = 3,
                 latent_dim: int = 256, init_temperature: float = 10.0,
                 n_codebooks: int = 8, codebook_size: int = 1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size

        self.anatomical_attention = AnatomicalAttention(
            in_channels=in_channels,
            num_organs=num_organs,
            init_temperature=init_temperature,
        )
        self.encoder = AnatomicalEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            num_organs=num_organs,
        )
        self.quantizer = ResidualVectorQuantizer(
            latent_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
        )
        self.decoder = AnatomicalDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            num_organs=num_organs,
        )

        # Optional: freeze gradients through the attention module
        self.freeze_attention_grad = False

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: Input image tensor (B, C, H, W).

        Returns:
            Dictionary with keys:
                - reconstructed: Reconstructed image (B, C, H, W).
                - organ_maps: Organ masks from attention (B, K, H, W).
                - vq_loss: Commitment loss (scalar).
                - latent: Quantized latent (B, latent_dim, H/16, W/16).
                - indices: Hard codebook indices (B, n_codebooks, H_lat*W_lat).
                - soft_probs: Soft assignment probabilities
                  (B, n_codebooks, H_lat*W_lat, codebook_size).
                - temp_fine: Temperature of fine prior generator.
                - temp_coarse: Temperature of coarse prior generator.
        """
        # 1. Anatomical attention
        attended_x, organ_maps = self.anatomical_attention(x)

        # Extract temperatures for logging
        if hasattr(self.anatomical_attention, 'prior_generator'):
            temp_fine = self.anatomical_attention.prior_generator.temperature
            temp_coarse = (
                self.anatomical_attention.prior_generator_coarse.temperature
            )
        else:
            temp_fine = temp_coarse = torch.tensor(0.0, device=x.device)

        if self.freeze_attention_grad:
            attended_x = attended_x.detach()
            organ_maps = organ_maps.detach()

        # 2. Encode
        latent = self.encoder(attended_x, organ_maps)   # (B, C_lat, H_lat, W_lat)
        B, _, H_lat, W_lat = latent.shape

        # 3. Quantize
        quantized, vq_loss, indices, soft_probs = self.quantizer(latent)

        # Reshape indices: (B*H_lat*W_lat, n_codebooks) → (B, n_codebooks, H_lat*W_lat)
        indices = (
            indices
            .view(B, H_lat * W_lat, self.n_codebooks)
            .permute(0, 2, 1)
            .contiguous()
        )

        # 4. Decode
        reconstructed = self.decoder(quantized, organ_maps)

        return {
            'reconstructed': reconstructed,
            'organ_maps':    organ_maps,
            'vq_loss':       vq_loss,
            'latent':        quantized,
            'indices':       indices,
            'soft_probs':    soft_probs,
            'temp_fine':     temp_fine,
            'temp_coarse':   temp_coarse,
        }
