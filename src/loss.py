"""
Loss function for medical image compression with diagnostic preservation.

Components:
    - ROI‑weighted MSE reconstruction loss (via differentiable foreground mask)
    - MS‑SSIM loss (warm‑up schedule)
    - Rate loss (soft BPP via DiscreteEntropyEstimator)
    - VQ commitment loss
    - LPIPS perceptual loss (warm‑up schedule)
    - Sparsity loss on organ maps (L1)
    - Foreground alignment loss (MSE between max organ map and foreground mask)
    - Map diversity loss (minimise cosine similarity between different maps)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .entropy import DiscreteEntropyEstimator


class MedicalCompressionLoss(nn.Module):
    """Composite loss for the anatomical attention compression model.

    The total loss is a weighted sum of reconstruction (MSE + MS‑SSIM + LPIPS),
    rate, VQ commitment, and regularisation terms (sparsity, foreground alignment,
    map diversity). Several components use epoch‑based warm‑up schedules.

    Args:
        lambda_rate: Weight for rate loss (BPP).
        lambda_vq: Weight for VQ commitment loss.
        lambda_perceptual: Weight for LPIPS perceptual loss.
        lambda_ssim: Weight for MS‑SSIM loss.
        lambda_sparsity: Weight for L1 sparsity loss on organ maps.
        lambda_roi: Weight for ROI‑weighted MSE (foreground emphasis).
        lambda_temperature: Weight for temperature regularisation (unused).
        lambda_diversity: Weight for map diversity loss.
        lambda_fg_align: Weight for foreground alignment loss.
        fg_align_warmup: Epoch after which foreground alignment loss activates.
        codebook_size: Number of entries per codebook.
        n_codebooks: Number of residual codebooks.
        perceptual_warmup_epochs: Epoch after which LPIPS loss activates.
        ssim_warmup_epochs: Epoch after which MS‑SSIM loss activates.
    """

    def __init__(
        self,
        lambda_rate:        float = 0.005,
        lambda_vq:          float = 0.25,
        lambda_perceptual:  float = 0.005,
        lambda_ssim:        float = 0.15,
        lambda_sparsity:    float = 0.0005,
        lambda_roi:         float = 0.1,
        lambda_temperature: float = 0.001,
        lambda_diversity:   float = 0.5,
        lambda_fg_align:    float = 0.5,
        fg_align_warmup:    int   = 3,
        codebook_size:      int   = 1024,
        n_codebooks:        int   = 8,
        perceptual_warmup_epochs: int = 30,
        ssim_warmup_epochs:       int = 10,
    ):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.lambda_vq = lambda_vq
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_sparsity = lambda_sparsity
        self.lambda_roi = lambda_roi
        self.lambda_temperature = lambda_temperature
        self.lambda_diversity = lambda_diversity
        self.lambda_fg_align = lambda_fg_align
        self.fg_align_warmup = fg_align_warmup
        self.perceptual_warmup_epochs = perceptual_warmup_epochs
        self.ssim_warmup_epochs = ssim_warmup_epochs
        self.current_epoch = 0

        self.mse_loss = nn.MSELoss()
        self.entropy_estimator = DiscreteEntropyEstimator(
            codebook_size=codebook_size,
            n_codebooks=n_codebooks,
        )
        self._lpips_model = None

    # ------------------------------------------------------------------
    def _get_lpips(self, device):
        """Lazily load the LPIPS model."""
        if self._lpips_model is None:
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net='alex').to(device)
                for p in self._lpips_model.parameters():
                    p.requires_grad_(False)
                print("✅ LPIPS perceptual model loaded.")
            except Exception as e:
                print(f"⚠️  LPIPS unavailable ({e}). Perceptual loss disabled.")
                self._lpips_model = "unavailable"
        return self._lpips_model

    # ------------------------------------------------------------------
    @staticmethod
    def _ms_ssim_loss(img1, img2):
        """Compute 1 - MS‑SSIM as a loss."""
        try:
            from pytorch_msssim import ms_ssim
            return 1.0 - ms_ssim(img1, img2, data_range=1.0, size_average=True)
        except ImportError:
            pass
        try:
            from pytorch_msssim import ssim
            return 1.0 - ssim(img1, img2, data_range=1.0, size_average=True)
        except ImportError:
            pass
        # Manual SSIM fallback
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu1 = F.avg_pool2d(img1, 11, 1, 0)
        mu2 = F.avg_pool2d(img2, 11, 1, 0)
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        sigma1_sq = F.avg_pool2d(img1 * img1, 11, 1, 0) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 11, 1, 0) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 11, 1, 0) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()

    @staticmethod
    def _soft_fg_mask(target: torch.Tensor) -> torch.Tensor:
        """Differentiable foreground mask using local variance.

        Args:
            target: Image tensor (B, C, H, W) in [0, 1].

        Returns:
            Foreground mask (B, 1, H, W), higher in structured regions.
        """
        gray = target.mean(dim=1, keepdim=True)
        kernel = 11
        pad = kernel // 2
        mu = F.avg_pool2d(gray, kernel, stride=1, padding=pad)
        mu2 = F.avg_pool2d(gray**2, kernel, stride=1, padding=pad)
        local_var = (mu2 - mu**2).clamp(min=0)

        # Normalise per image to [0,1]
        B = local_var.shape[0]
        flat = local_var.view(B, -1)
        vmin = flat.min(dim=1).values.view(B, 1, 1, 1)
        vmax = flat.max(dim=1).values.view(B, 1, 1, 1) + 1e-8
        fg_soft = (local_var - vmin) / (vmax - vmin)
        return fg_soft

    # ------------------------------------------------------------------
    def forward(self, output: dict, target: torch.Tensor) -> dict:
        """Compute all loss components.

        Args:
            output: Dictionary from MedicalCompressor.forward().
            target: Ground truth image (B, C, H, W) in [0, 1].

        Returns:
            Dictionary with total_loss and individual components
            (mse_loss, ssim_loss, rate_loss, vq_loss, perceptual_loss,
             sparsity_loss, fg_align_loss, diversity_loss, psnr, bpp, etc.).
        """
        recon = output['reconstructed']
        indices = output['indices']
        organ_maps = output.get('organ_maps', None)
        vq_loss = output['vq_loss']
        H, W = target.shape[-2:]

        # 1. ROI‑weighted MSE
        fg_soft = self._soft_fg_mask(target).detach()
        if self.lambda_roi > 0:
            roi_weight = 1.0 + self.lambda_roi * fg_soft
            rec_loss = ((recon - target) ** 2 * roi_weight).mean()
        else:
            rec_loss = self.mse_loss(recon, target)

        # 2. MS‑SSIM (with warm‑up)
        ssim_loss = torch.tensor(0.0, device=recon.device)
        if self.current_epoch >= self.ssim_warmup_epochs and self.lambda_ssim > 0.0:
            ssim_loss = self._ms_ssim_loss(recon, target)

        # 3. Rate loss
        soft_probs = output['soft_probs']
        bpp = self.entropy_estimator.soft_bpp(soft_probs, image_hw=(H, W))
        rate_loss = bpp

        bpp_indices = self.entropy_estimator(indices, image_hw=(H, W))

        # 4. VQ commitment loss (already provided)

        # 5. Perceptual loss (with warm‑up)
        perc_loss = torch.tensor(0.0, device=recon.device)
        if (self.current_epoch >= self.perceptual_warmup_epochs
                and self.lambda_perceptual > 0.0):
            lpips_model = self._get_lpips(recon.device)
            if lpips_model not in (None, "unavailable"):
                recon_norm = recon * 2.0 - 1.0
                target_norm = target * 2.0 - 1.0
                perc_loss = lpips_model(recon_norm, target_norm).mean()

        # 6. Sparsity loss
        if organ_maps is not None:
            sparsity_loss = torch.mean(torch.abs(organ_maps))
        else:
            sparsity_loss = torch.tensor(0.0, device=recon.device)

        # 7. Foreground alignment loss
        fg_align_loss = torch.tensor(0.0, device=recon.device)
        if organ_maps is not None and self.current_epoch >= self.fg_align_warmup:
            fg_soft_detached = self._soft_fg_mask(target).detach()
            max_map = organ_maps.max(dim=1, keepdim=True).values
            fg_align_loss = F.mse_loss(max_map, fg_soft_detached)

        # 8. Map diversity loss
        diversity_loss = torch.tensor(0.0, device=recon.device)
        if organ_maps is not None and organ_maps.shape[1] > 1:
            B, K, H_map, W_map = organ_maps.shape
            flat_maps = organ_maps.view(B, K, -1)
            flat_maps_norm = F.normalize(flat_maps, p=2, dim=2)
            sim_matrix = torch.bmm(flat_maps_norm,
                                   flat_maps_norm.transpose(1, 2))
            mask_off = ~torch.eye(K, dtype=torch.bool, device=recon.device)
            diversity_loss = sim_matrix[:, mask_off].mean()

        # Total loss
        total_loss = (
            rec_loss
            + self.lambda_ssim * ssim_loss
            + self.lambda_rate * rate_loss
            + self.lambda_vq * vq_loss
            + self.lambda_perceptual * perc_loss
            + self.lambda_sparsity * sparsity_loss
            + self.lambda_fg_align * fg_align_loss
            + self.lambda_diversity * diversity_loss
        )

        psnr = 10 * torch.log10(1.0 / (rec_loss + 1e-8))

        return {
            'total_loss': total_loss,
            'mse_loss': rec_loss,
            'rec_loss': rec_loss,
            'ssim_loss': ssim_loss,
            'rate_loss': rate_loss,
            'vq_loss': vq_loss,
            'perceptual_loss': perc_loss,
            'sparsity_loss': sparsity_loss,
            'fg_align_loss': fg_align_loss,
            'diversity_loss': diversity_loss,
            'psnr': psnr,
            'bpp': bpp,
            'bpp_indices': bpp_indices,
            'rate_loss_indices': bpp_indices,
        }
