"""
Residual vector quantizer with exponential moving average (EMA) updates.

Implements a multi‑codebook VQ layer where each codebook quantises the
residual of the previous one. Uses EMA updates for codebook vectors and
includes a dead‑code reset mechanism to maintain codebook utilisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    """Multi‑codebook residual vector quantizer.

    Quantises latent vectors sequentially across `n_codebooks`, where each
    codebook captures the residual left by the preceding ones. The forward
    pass returns the quantized latent, commitment loss, hard indices, and
    soft assignment probabilities (for differentiable entropy estimation).

    Args:
        latent_dim: Dimensionality of the latent vectors.
        n_codebooks: Number of residual codebooks.
        codebook_size: Number of entries per codebook.
        commitment_weight: Weight applied to the commitment loss.
        ema_decay: Decay factor for EMA updates of codebook vectors.
    """

    def __init__(self, latent_dim: int = 256, n_codebooks: int = 8,
                 codebook_size: int = 1024, commitment_weight: float = 1.0,
                 ema_decay: float = 0.99):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay

        # Temperature for soft assignment during training
        self.log_soft_temperature = nn.Parameter(torch.tensor(-2.0))

        # Codebook embeddings
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, latent_dim) for _ in range(n_codebooks)
        ])

        # Linear projections between codebooks (except the last)
        self.residual_proj = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(n_codebooks - 1)
        ])

        # Initialise codebooks and projections
        for cb in self.codebooks:
            nn.init.xavier_uniform_(cb.weight, gain=0.5)
        for proj in self.residual_proj:
            nn.init.xavier_uniform_(proj.weight, gain=0.5)
            nn.init.zeros_(proj.bias)

        # EMA buffers
        for i, cb in enumerate(self.codebooks):
            self.register_buffer(f'ema_cluster_size_{i}',
                                 torch.zeros(codebook_size))
            self.register_buffer(f'ema_embed_sum_{i}',
                                 cb.weight.data.clone())

    @property
    def soft_temperature(self) -> torch.Tensor:
        """Exponentiated temperature, clamped for numerical stability."""
        return torch.exp(self.log_soft_temperature).clamp(min=1e-6)

    def _ema_update(self, cb_idx: int, flat_z: torch.Tensor,
                    indices: torch.Tensor):
        """Update codebook vectors and dead codes using EMA."""
        if not self.training:
            return
        cb = self.codebooks[cb_idx]
        cluster_size = getattr(self, f'ema_cluster_size_{cb_idx}')
        embed_sum = getattr(self, f'ema_embed_sum_{cb_idx}')

        # EMA update of cluster sizes and embedding sums
        one_hot = F.one_hot(indices, self.codebook_size).float()
        new_cluster_size = one_hot.sum(0)
        cluster_size.data.mul_(self.ema_decay).add_(
            new_cluster_size, alpha=1 - self.ema_decay
        )
        new_embed_sum = one_hot.T @ flat_z
        embed_sum.data.mul_(self.ema_decay).add_(
            new_embed_sum, alpha=1 - self.ema_decay
        )

        # Laplace smoothing of cluster sizes
        n = cluster_size.sum()
        smoothed = (cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
        cb.weight.data.copy_(embed_sum / smoothed.unsqueeze(1))

        # Reset dead codes (cluster size < 1)
        dead = (cluster_size < 1.0)
        n_dead = dead.sum().item()
        if n_dead > 0 and flat_z.shape[0] > 0:
            n_sample = min(n_dead, flat_z.shape[0])
            perm = torch.randperm(flat_z.shape[0], device=flat_z.device)[:n_sample]
            replacement = flat_z[perm].detach()
            dead_indices = dead.nonzero(as_tuple=False).squeeze(1)[:n_sample]
            cb.weight.data[dead_indices] = replacement
            cluster_size[dead_indices] = 1.0
            embed_sum[dead_indices] = replacement

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor]:
        """Quantize latent vectors.

        Args:
            z: Latent tensor (B, C, H, W).

        Returns:
            Tuple containing:
                - quantized: Quantized latent (B, C, H, W).
                - vq_loss: Commitment loss (scalar).
                - indices: Hard codebook indices (B * H * W, n_codebooks).
                - soft_probs: Soft assignment probabilities
                  (B, n_codebooks, H*W, codebook_size).
        """
        B, C, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1)          # (B, H, W, C)
        z_flat = z_perm.reshape(-1, C)           # (B*H*W, C)

        residual_attached = z_flat
        residual_detached = z_flat.detach()

        quantized_total = torch.zeros_like(z_flat)
        all_indices = []
        all_soft_probs = []
        commitment_loss = 0.0

        for i, cb in enumerate(self.codebooks):
            # Hard assignment
            distances = torch.cdist(residual_detached, cb.weight)
            indices = torch.argmin(distances, dim=1)   # (B*H*W,)
            quantized_i = cb(indices)                  # (B*H*W, C)

            # Soft assignment (for entropy estimation)
            distances_soft = torch.cdist(residual_attached, cb.weight.detach())
            soft_probs = F.softmax(
                -distances_soft / (self.soft_temperature + 1e-6), dim=-1
            )
            all_soft_probs.append(soft_probs)

            # EMA update of codebook
            self._ema_update(i, residual_detached, indices)

            # Straight‑through estimator
            quantized_i_st = (residual_attached +
                              (quantized_i - residual_attached).detach())
            quantized_total += quantized_i_st

            # Commitment loss
            commitment_loss += F.mse_loss(residual_attached,
                                          quantized_i.detach())

            # Compute residual for next codebook
            residual_attached = residual_attached - quantized_i
            residual_detached = residual_detached - quantized_i.detach()

            if i < self.n_codebooks - 1:
                residual_attached = self.residual_proj[i](residual_attached)
                residual_detached = self.residual_proj[i](
                    residual_detached
                ).detach()

            all_indices.append(indices)

        # Reshape outputs
        quantized_out = quantized_total.reshape(B, H, W, C).permute(0, 3, 1, 2)
        indices_stack = torch.stack(all_indices, dim=1)   # (B*H*W, n_cb)
        total_vq_loss = self.commitment_weight * commitment_loss

        soft_probs_stack = torch.stack(all_soft_probs, dim=0)  # (n_cb, B*H*W, cs)
        soft_probs_stack = soft_probs_stack.view(
            self.n_codebooks, B, H * W, self.codebook_size
        ).permute(1, 0, 2, 3).contiguous()  # (B, n_cb, H*W, codebook_size)

        return quantized_out, total_vq_loss, indices_stack, soft_probs_stack
