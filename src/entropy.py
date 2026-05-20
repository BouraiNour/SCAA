"""
Differentiable entropy estimation for vector‑quantized representations.

Provides a discrete entropy estimator that computes bits‑per‑pixel (BPP)
from either hard codebook indices (forward pass) or soft assignment
probabilities (soft_bpp), using learned prior logits per codebook.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteEntropyEstimator(nn.Module):
    """Learned prior for entropy estimation of VQ codebook indices.

    Maintains a set of learnable logits (one per codebook) that define a
    categorical prior over codebook entries. The BPP is computed as the
    cross‑entropy between the empirical distribution of indices (or soft
    assignments) and the learned prior.

    Args:
        codebook_size: Number of entries per codebook.
        n_codebooks: Number of residual codebooks.
        temperature: Softmax temperature for the prior logits.
    """

    def __init__(self, codebook_size: int = 1024, n_codebooks: int = 8,
                 temperature: float = 0.1):
        super().__init__()
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        self.temperature = temperature
        self.prior_logits = nn.Parameter(
            torch.zeros(n_codebooks, codebook_size)
        )

    def soft_bpp(self, soft_probs: torch.Tensor,
                 image_hw: tuple = (256, 256)) -> torch.Tensor:
        """Compute differentiable BPP from soft assignment probabilities.

        Args:
            soft_probs: Tensor of shape (B, n_codebooks, N, codebook_size)
                where N = H_lat * W_lat.
            image_hw: Tuple (H, W) of the original image.

        Returns:
            Bits per pixel as a scalar tensor.
        """
        B, n_cb, N, cb_size = soft_probs.shape
        H, W = image_hw
        _log2 = math.log(2.0)

        # Combine batch and spatial: (B*N, n_cb, cb_size)
        soft_probs_flat = soft_probs.permute(1, 0, 2, 3)  # (n_cb, B, N, cb_size)
        soft_probs_flat = soft_probs_flat.reshape(n_cb, -1, cb_size)
        soft_probs_flat = soft_probs_flat.permute(1, 0, 2)  # (B*N, n_cb, cb_size)

        total_bits = 0.0
        for cb_idx in range(n_cb):
            probs = soft_probs_flat[:, cb_idx, :]   # (B*N, cb_size)
            empirical_counts = probs.sum(dim=0).detach()
            empirical_probs = empirical_counts / (empirical_counts.sum() + 1e-8)

            log_prior = F.log_softmax(
                self.prior_logits[cb_idx] / self.temperature, dim=0
            )
            log_prior = log_prior.to(empirical_probs.device)

            bits_per_symbol = -(empirical_probs * log_prior / _log2).sum()
            total_bits += bits_per_symbol * (B * N)

        bpp = total_bits / (B * H * W)
        return bpp

    def forward(self, indices: torch.Tensor,
                image_hw: tuple = (256, 256)) -> torch.Tensor:
        """Compute BPP from hard codebook indices (non‑differentiable).

        Args:
            indices: Tensor of shape (B, n_codebooks, N) with integer indices.
            image_hw: Tuple (H, W) of the original image.

        Returns:
            Bits per pixel as a scalar tensor.
        """
        B, n_cb, N = indices.shape
        H, W = image_hw
        _log2 = math.log(2.0)

        total_bits = torch.tensor(0.0, device=self.prior_logits.device)

        for cb_idx in range(n_cb):
            idx = indices[:, cb_idx, :].reshape(-1)
            one_hot = F.one_hot(idx, self.codebook_size).float().to(
                self.prior_logits.device
            )
            log_prior = F.log_softmax(
                self.prior_logits[cb_idx] / self.temperature, dim=0
            )

            empirical_counts = one_hot.sum(0)
            empirical_probs = empirical_counts / empirical_counts.sum()

            bits_per_symbol = -(empirical_probs * log_prior / _log2).sum()
            total_bits = total_bits + bits_per_symbol * B * N

        bpp = total_bits / (B * H * W)
        return bpp
