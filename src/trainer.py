"""
Training loop for the medical image compression model.

Provides MedicalCompressionTrainer, which handles:
    - Model, loss, optimizer, and scheduler setup.
    - DataLoader construction via `build_loaders`.
    - Epoch‑level training and validation with logging.
    - Checkpoint saving and resumption.
    - Integration with DiagnosticEvaluator for diagnostic metrics.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Dict, Optional, Tuple

from .dataset import build_loaders
from .model.compressor import MedicalCompressor
from .loss import MedicalCompressionLoss
from .evaluator import DiagnosticEvaluator


class MedicalCompressionTrainer:
    """Trainer for the anatomical‑attention compression model.

    Args:
        config: Dictionary with training configuration.
        output_dir: Directory for checkpoints, logs, and visualizations.
    """

    def __init__(self, config: dict, output_dir: str):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Model
        self.model = MedicalCompressor(
            in_channels=config['in_channels'],
            num_organs=config['num_organs'],
            latent_dim=config.get('latent_dim', 256),
            init_temperature=config.get('init_temperature', 10.0),
            n_codebooks=config.get('n_codebooks', 8),
            codebook_size=config.get('codebook_size', 1024),
        ).to(self.device)

        # Freeze temperature parameters initially
        for param_name, param in self.model.named_parameters():
            if 'temperature' in param_name:
                param.requires_grad_(False)

        # Ablation: replace AnatomicalAttention with IdentityAttention if needed
        if not config.get('use_anatomical_attention', True):
            self.model.anatomical_attention = IdentityAttention(
                in_channels=config['in_channels'],
                num_organs=config['num_organs'],
            ).to(self.device)
            print("⚠️  ABLATION: AnatomicalAttention replaced with IdentityAttention")
        else:
            print("✅ Using full AnatomicalAttention")

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params / 1e6:.2f}M")

        # Diagnostic evaluator
        self.diagnostic_eval = DiagnosticEvaluator(
            masks_dir=config.get('masks_dir', ''),
            device=str(self.device),
            save_dir=config.get('diag_save_dir', 'diagnostic_results'),
            latent_hw=config.get('latent_hw', (16, 16)),
            image_hw=config.get('image_size', (256, 256)),
            patch_size=config.get('patch_size', 256),
        )

        # Loss
        self.criterion = MedicalCompressionLoss(
            lambda_rate=config.get('lambda_rate', 0.005),
            lambda_vq=config.get('lambda_vq', 0.25),
            lambda_perceptual=config.get('lambda_perceptual', 0.005),
            lambda_ssim=config.get('lambda_ssim', 0.15),
            lambda_sparsity=config.get('lambda_sparsity', 0.0005),
            lambda_roi=config.get('lambda_roi', 0.1),
            lambda_diversity=config.get('lambda_diversity', 0.5),
            lambda_fg_align=config.get('lambda_fg_align', 0.5),
            fg_align_warmup=config.get('fg_align_warmup', 3),
            codebook_size=config.get('codebook_size', 1024),
            n_codebooks=config.get('n_codebooks', 8),
            perceptual_warmup_epochs=config.get('perceptual_warmup', 30),
            ssim_warmup_epochs=config.get('ssim_warmup', 10),
        ).to(self.device)

        # Optimizer (separate weight decay groups)
        decay_params, no_decay_params = [], []
        for cb in self.model.quantizer.codebooks:
            cb.weight.requires_grad_(False)

        for name, p in list(self.model.named_parameters()) + list(self.criterion.named_parameters()):
            if not p.requires_grad:
                continue
            if p.ndim < 2 or 'prior_logits' in name:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        self.optimizer = optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': config.get('weight_decay', 1e-4)},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=config.get('learning_rate', 1e-4),
        )

        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.temperature_unfreeze_epoch = config.get('temperature_unfreeze_epoch', 5)
        _cosine_epochs = config.get('epochs', 150) - self.warmup_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(_cosine_epochs, 1),
            eta_min=config.get('learning_rate', 1e-4) / 100,
        )

        # Metrics storage
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.best_val_psnr = 0.0

        self.create_directories()

    def create_directories(self):
        """Create output directories for checkpoints, logs, visualizations."""
        for d in ['checkpoints', 'logs', 'visualizations']:
            os.makedirs(d, exist_ok=True)

    def setup_data_loaders(self):
        """Build train/val/test DataLoaders."""
        result = build_loaders(
            data_dir=self.config.get('data_dir'),
            batch_size=self.config['batch_size'],
            modality=self.config.get('modality', 'ct'),
            patch_size=self.config.get('patch_size', 256),
            val_frac=self.config.get('val_frac', 0.15),
            test_frac=self.config.get('test_frac', 0.15),
            num_workers=self.config.get('num_workers', 4),
            seed=42,
            split_save_path=self.config.get('split_save_path', 'split_indices.json'),
            metadata_csv=self.config.get('metadata_csv'),
        )
        if len(result) == 3:
            self.train_loader, self.val_loader, self.test_loader = result
        else:
            self.train_loader, self.val_loader = result
            self.test_loader = None

    def _get_lr_scale(self, epoch: int) -> float:
        """Linear warmup factor for the learning rate."""
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        return 1.0

    def _apply_lr_warmup(self, epoch: int):
        """Adjust learning rate during warmup."""
        scale = self._get_lr_scale(epoch)
        base_lr = self.config.get('learning_rate', 1e-4)
        for group in self.optimizer.param_groups:
            group['lr'] = base_lr * scale

    def _unfreeze_temperature(self, epoch: int):
        """Enable gradient for temperature parameters after a given epoch."""
        if epoch != self.temperature_unfreeze_epoch:
            return
        temp_params = []
        for param_name, param in self.model.named_parameters():
            if 'temperature' in param_name:
                param.requires_grad_(True)
                param.data.add_(0.01)
                temp_params.append(param)
        if temp_params:
            self.optimizer.add_param_group({
                'params': temp_params,
                'lr': self.config.get('learning_rate', 1e-4),
                'weight_decay': 0.0,
            })
            print(f"🔥 Temperature parameters unfrozen at epoch {epoch}.")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of averaged loss components.
        """
        self.model.train()
        epoch_metrics = defaultdict(float)

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(images)
            losses = self.criterion(output, images)

            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.criterion.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            for key, value in losses.items():
                epoch_metrics[key] += (value.item() if isinstance(value, torch.Tensor) else value)

            if batch_idx % self.config['log_interval'] == 0:
                lr_now = self.optimizer.param_groups[0]['lr']
                fg_align_val = losses.get('fg_align_loss', torch.tensor(0.0)).item()
                print(f'Train Epoch: {epoch} '
                      f'[{batch_idx * len(images)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)] '
                      f'Loss: {losses["total_loss"].item():.6f} '
                      f'PSNR: {losses["psnr"].item():.2f} dB '
                      f'FGAlign: {fg_align_val:.4f} '
                      f'LR: {lr_now:.2e}')

        for key in epoch_metrics:
            epoch_metrics[key] /= len(self.train_loader)
            self.train_metrics[key].append(epoch_metrics[key])

        return epoch_metrics

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one validation epoch with diagnostic metrics.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of averaged validation metrics.
        """
        self.model.eval()
        val_metrics = defaultdict(float)
        save_figures = (epoch % self.config.get('viz_interval', 5) == 0)

        for batch_idx, batch in enumerate(self.val_loader):
            images = batch['image'].to(self.device)
            filenames = batch['filename']
            patch_rows = batch['patch_row'].tolist()
            patch_cols = batch['patch_col'].tolist()

            output = self.model(images)
            losses = self.criterion(output, images)
            for k, v in losses.items():
                val_metrics[k] += v.item() if isinstance(v, torch.Tensor) else v

            # Bitrate map
            brt_map = self.diagnostic_eval.compute_spatial_bitrate_map(
                indices=output['indices'],
                latent_hw=self.diagnostic_eval.latent_hw,
                codebook_size=self.model.codebook_size,
                image_hw=self.diagnostic_eval.image_hw,
            )

            # ROI metrics
            roi = self.diagnostic_eval.compute_roi_metrics(
                output['reconstructed'], images, filenames,
                patch_rows, patch_cols,
            )
            # Organ map metrics
            omap = self.diagnostic_eval.compute_organ_map_metrics(
                output['organ_maps'], images, filenames,
                patch_rows, patch_cols,
            )
            # Bitrate-organ alignment
            align = self.diagnostic_eval.bitrate_organ_alignment(
                brt_map, output['organ_maps'],
            )
            # Error alignment
            err_ali = self.diagnostic_eval.error_organ_alignment(
                images, output['reconstructed'], output['organ_maps'],
                filenames, patch_rows, patch_cols,
            )

            # Temperatures
            if hasattr(self.model.anatomical_attention, 'prior_generator'):
                temp_fine = self.model.anatomical_attention.prior_generator.temperature.item()
                temp_coarse = self.model.anatomical_attention.prior_generator_coarse.temperature.item()
            else:
                temp_fine = temp_coarse = 0.0
            val_metrics['temp_fine'] += temp_fine
            val_metrics['temp_coarse'] += temp_coarse

            # True BPP
            image_size = self.config.get('image_size', (256, 256))
            true_bpp = self.compute_true_bpp(output['indices'], original_size=image_size)
            bpp_est = losses.get('bpp', torch.tensor(0.0)).item()

            batch_metrics = {
                'true_bpp': true_bpp,
                'true_compression_ratio': 16.0 / (true_bpp + 1e-8),
                'bpp_gap': abs(bpp_est - true_bpp),
                **roi,
                **omap,
                **align,
                **err_ali,
            }
            for key, value in batch_metrics.items():
                if isinstance(value, torch.Tensor):
                    val_metrics[key] += value.item()
                else:
                    val_metrics[key] += value

            if save_figures and batch_idx == 0:
                self.diagnostic_eval.save_paper_figure(
                    original=images,
                    reconstructed=output['reconstructed'],
                    organ_maps=output['organ_maps'],
                    bitrate_map=brt_map,
                    filenames=filenames,
                    patch_rows=patch_rows,
                    patch_cols=patch_cols,
                    epoch=epoch,
                    save_subdir='progress',
                    batch_idx=batch_idx,
                    n_images=2,
                )

        # Average
        for key in val_metrics:
            val_metrics[key] /= len(self.val_loader)
            self.val_metrics[key].append(val_metrics[key])

        val_metrics['ssim'] = 1.0 - val_metrics.get('ssim_loss', 0.0)
        val_metrics['lpips'] = val_metrics.get('perceptual_loss', 0.0)

        self.print_validation_metrics(epoch, val_metrics)

        if val_metrics['psnr'] > self.best_val_psnr:
            self.best_val_psnr = val_metrics['psnr']
            self.save_checkpoint(epoch, is_best=True)

        return val_metrics

    def compute_true_bpp(self, indices: torch.Tensor,
                         original_size: tuple = (256, 256)) -> float:
        """Compute empirical entropy BPP from codebook indices.

        Args:
            indices: Tensor of shape (B, n_codebooks, N).
            original_size: (H, W) of the original image.

        Returns:
            True bits per pixel.
        """
        B, n_codebooks, N = indices.shape
        total_bits = 0
        for cb in range(n_codebooks):
            idx = indices[:, cb, :].reshape(-1).cpu().numpy()
            unique, counts = np.unique(idx, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            total_bits += entropy * len(idx)
        return total_bits / (B * original_size[0] * original_size[1])

    def print_validation_metrics(self, epoch: int, metrics: dict):
        """Print a formatted summary of validation metrics."""
        print(f"\n{'='*80}")
        print(f"VALIDATION EPOCH {epoch}")
        print(f"{'='*80}")
        print(f"\n📊 Quality:")
        print(f"   PSNR:               {metrics.get('psnr',0):7.2f} dB")
        print(f"   MS‑SSIM loss:        {metrics.get('ssim_loss',0):7.4f}")
        print(f"   LPIPS:               {metrics.get('perceptual_loss',0):7.4f}")
        print(f"\n📈 Entropy BPP (soft):  {metrics.get('bpp',0):7.4f}")
        print(f"   True BPP:            {metrics.get('true_bpp',0):7.4f}")
        print(f"   BPP Gap:             {metrics.get('bpp_gap',0):7.4f}")
        print(f"\n🎯 Loss Components:")
        print(f"   Total:               {metrics.get('total_loss',0):7.4f}")
        print(f"   MSE:                 {metrics.get('mse_loss',0):7.4f}")
        print(f"   SSIM:                {metrics.get('ssim_loss',0):7.4f}")
        print(f"   Rate:                {metrics.get('rate_loss',0):7.4f}")
        print(f"   VQ:                  {metrics.get('vq_loss',0):7.4f}")
        print(f"   Perceptual:          {metrics.get('perceptual_loss',0):7.4f}")
        print(f"   Sparsity:            {metrics.get('sparsity_loss',0):7.4f}")
        print(f"   FG Align:            {metrics.get('fg_align_loss',0):7.4f}")
        print(f"\n🔬 Diagnostic Preservation:")
        print(f"   ROI PSNR:            {metrics.get('roi_psnr',0):7.2f} dB")
        print(f"   BG PSNR:             {metrics.get('bg_psnr',0):7.2f} dB")
        print(f"   ROI Gain:            {metrics.get('roi_psnr_gain',0):+7.2f} dB")
        print(f"   Map FG Alignment:    {metrics.get('map_fg_alignment',0):7.3f}")
        print(f"   Map Diversity:       {metrics.get('map_inter_diversity',0):7.3f}")
        print(f"   Bitrate‑Organ Corr:  {metrics.get('bitrate_organ_corr',0):6.3f}")
        print(f"   Bitrate FG Mass:     {metrics.get('bitrate_fg_mass',0):6.3f}")
        print(f"   ROI/BG Error Ratio:  {metrics.get('roi_error_ratio',1):6.3f}")
        print(f"   Temp Fine/Coarse:    {metrics.get('temp_fine',0):.2f} / "
              f"{metrics.get('temp_coarse',0):.2f}")
        print(f"{'='*80}\n")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model and training state to disk.

        Args:
            epoch: Current epoch.
            is_best: If True, also saves as 'best_model.pth'.
        """
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_psnr': self.best_val_psnr,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'config': self.config,
        }
        drive_dir = self.config.get('drive_checkpoint_dir', './checkpoints')
        os.makedirs(drive_dir, exist_ok=True)

        latest_path = os.path.join(drive_dir, 'latest_model.pth')
        torch.save(ckpt, latest_path)
        print(f"💾 Latest checkpoint saved to {latest_path} (epoch {epoch})")

        if is_best:
            best_path = os.path.join(drive_dir, 'best_model.pth')
            torch.save(ckpt, best_path)
            print(f"🏆 Best model saved to {best_path} (PSNR: {self.best_val_psnr:.2f} dB)")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load a checkpoint and return the epoch to resume from.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            The epoch number saved in the checkpoint.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.best_val_psnr = ckpt['best_val_psnr']
        self.train_metrics.update(ckpt['train_metrics'])
        self.val_metrics.update(ckpt['val_metrics'])
        print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']} "
              f"(best PSNR so far: {self.best_val_psnr:.2f} dB)")
        return ckpt['epoch']

    def save_metrics_to_json(self):
        """Save training and validation metrics as a JSON file."""
        with open('logs/training_metrics.json', 'w') as f:
            json.dump({
                'train_metrics': dict(self.train_metrics),
                'val_metrics': dict(self.val_metrics),
                'config': self.config,
            }, f, indent=2)

    def train(self, resume_from: Optional[str] = None):
        """Run the full training loop.

        Args:
            resume_from: Path to a checkpoint to resume training.
        """
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resumed from epoch {start_epoch}")

        self.setup_data_loaders()

        print(f"\n🚀 Starting training for {self.config['epochs']} epochs")
        print(f"   latent_dim={self.config.get('latent_dim',256)}, "
              f"n_codebooks={self.config.get('n_codebooks',8)}, "
              f"codebook_size={self.config.get('codebook_size',1024)}")

        for epoch in range(start_epoch, self.config['epochs']):
            epoch_start = time.time()

            if epoch < self.warmup_epochs:
                self._apply_lr_warmup(epoch)

            if epoch == self.temperature_unfreeze_epoch:
                self._unfreeze_temperature(epoch)

            self.criterion.current_epoch = epoch

            self.train_epoch(epoch)
            self.validate_epoch(epoch)

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            if epoch % self.config.get('checkpoint_interval', 5) == 0:
                self.save_checkpoint(epoch)

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"⏰ Epoch {epoch} completed in {epoch_time:.1f}s  |  LR: {current_lr:.2e}")

            self.save_metrics_to_json()

        # Final comprehensive evaluation
        print("\n🚀 Starting final comprehensive evaluation...")
        final_metrics = self.diagnostic_eval.run_full_evaluation(
            model=self.model,
            val_loader=self.val_loader,
            epoch=self.config['epochs'],
            label='Final_Evaluation',
            save_figures=True,
        )
        final_metrics_path = f'logs/final_metrics_epoch_{self.config["epochs"]}.json'
        with open(final_metrics_path, 'w') as f:
            clean_metrics = {k: float(v) for k, v in final_metrics.items()}
            json.dump(clean_metrics, f, indent=2)
        print(f"📁 Final metrics saved to {final_metrics_path}")
        print("✅ Training complete!")
