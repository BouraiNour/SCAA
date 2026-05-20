"""
Diagnostic preservation evaluator for medical image compression.

Evaluates how well a compression model preserves diagnostically relevant
information by computing region‑of‑interest (ROI) PSNR, organ map quality,
bitrate–organ alignment, and reconstruction‑error alignment against
ground‑truth anatomical masks.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes


class DiagnosticEvaluator:
    """Evaluator that quantifies diagnostic preservation.

    Uses a 5‑level importance map:
        air = 0.0, soft_tissue = 0.3, mediastinum = 0.6,
        lung = 0.8, nodule = 1.0.

    Args:
        masks_dir: Directory containing *_lung_mask.npy and *_nodule_mask.npy files.
        device: Torch device.
        save_dir: Directory for output figures.
        latent_hw: Tuple (H_lat, W_lat) of the latent spatial size.
        image_hw: Tuple (H, W) of the input image.
        patch_size: Size of the square patches.
    """

    IMPORTANCE = {
        'air':         0.0,
        'soft_tissue': 0.3,
        'mediastinum': 0.6,
        'lung':        0.8,
        'nodule':      1.0,
    }

    def __init__(self, masks_dir: str, device: str = 'cuda',
                 save_dir: str = 'diagnostic_results',
                 latent_hw: tuple = (16, 16), image_hw: tuple = (256, 256),
                 patch_size: int = 256):
        self.masks_dir = masks_dir
        self.device = device
        self.save_dir = save_dir
        self.latent_hw = latent_hw
        self.image_hw = image_hw
        self.patch_size = patch_size

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/organ_maps', exist_ok=True)
        os.makedirs(f'{save_dir}/roi_overlays', exist_ok=True)
        os.makedirs(f'{save_dir}/paper_figures', exist_ok=True)

    # ------------------------------------------------------------------
    # Body mask helper
    # ------------------------------------------------------------------
    def _body_mask_np(self, ct_np: np.ndarray) -> np.ndarray:
        """Otsu threshold + hole‑filling on a normalised CT patch."""
        if ct_np.max() > ct_np.min():
            thresh = threshold_otsu(ct_np)
        else:
            return np.zeros_like(ct_np, dtype=np.uint8)
        body = (ct_np > thresh).astype(np.uint8)
        return binary_fill_holes(body).astype(np.uint8)

    # ------------------------------------------------------------------
    # Mask loader
    # ------------------------------------------------------------------
    def _load_mask_patch(self, mask_path: str, row: int, col: int,
                         patch_size: int) -> np.ndarray:
        """Load a whole‑slice mask and crop to the same patch as the slice.

        The mask is first binarized (value > 0), then resized (nearest
        neighbour) to match the slice’s target grid, then cropped to the
        correct patch.

        Args:
            mask_path: Path to the .npy mask file.
            row: Patch row index.
            col: Patch column index.
            patch_size: Edge length of the patch.

        Returns:
            Binary mask patch of shape (patch_size, patch_size).
        """
        mask = np.load(mask_path).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        mH, mW = mask.shape
        tH = max(patch_size, round(mH / patch_size) * patch_size)
        tW = max(patch_size, round(mW / patch_size) * patch_size)

        if (mH, mW) != (tH, tW):
            t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(tH, tW), mode='nearest')
            mask = t.squeeze().numpy()

        r0, r1 = row * patch_size, (row + 1) * patch_size
        c0, c1 = col * patch_size, (col + 1) * patch_size
        return mask[r0:r1, c0:c1]

    # ------------------------------------------------------------------
    # Diagnostic importance map
    # ------------------------------------------------------------------
    def build_importance_map(self, ct_slices: torch.Tensor,
                             filenames: list,
                             patch_rows: list = None,
                             patch_cols: list = None,
                             patch_size: int = None) -> torch.Tensor:
        """Create a per‑pixel importance map for a batch of patches.

        Args:
            ct_slices: (B, 1, H, W) normalised CT patch.
            filenames: List of original .npy slice names.
            patch_rows: List of patch row indices.
            patch_cols: List of patch column indices.
            patch_size: Override default patch size.

        Returns:
            Importance map (B, 1, H, W) with values in [0, 1].
        """
        ps = patch_size if patch_size is not None else self.patch_size
        B, _, H, W = ct_slices.shape
        imp_batch = []

        for b in range(B):
            stem = filenames[b].replace('.npy', '')
            row  = int(patch_rows[b]) if patch_rows is not None else 0
            col  = int(patch_cols[b]) if patch_cols is not None else 0

            lung_path   = os.path.join(self.masks_dir, f'{stem}_lung_mask.npy')
            nodule_path = os.path.join(self.masks_dir, f'{stem}_nodule_mask.npy')

            lung_mask   = self._load_mask_patch(lung_path,   row, col, ps)
            nodule_mask = self._load_mask_patch(nodule_path, row, col, ps)

            ct_np        = ct_slices[b, 0].cpu().numpy()
            body_mask    = self._body_mask_np(ct_np).astype(np.float32)
            mediastinum  = np.clip(body_mask - lung_mask, 0.0, 1.0)

            imp  = np.zeros((H, W), dtype=np.float32)
            imp += self.IMPORTANCE['soft_tissue']                                    * body_mask
            imp += (self.IMPORTANCE['mediastinum'] - self.IMPORTANCE['soft_tissue']) * mediastinum
            imp += (self.IMPORTANCE['lung']        - self.IMPORTANCE['mediastinum']) * lung_mask
            imp += (self.IMPORTANCE['nodule']      - self.IMPORTANCE['lung'])        * nodule_mask
            imp  = np.clip(imp, 0.0, 1.0)

            imp_batch.append(torch.from_numpy(imp).unsqueeze(0))

        return torch.stack(imp_batch, dim=0).to(self.device)

    def foreground_mask(self, ct_slices: torch.Tensor, filenames: list,
                        patch_rows: list = None, patch_cols: list = None,
                        patch_size: int = None,
                        threshold: float = 0.25) -> torch.Tensor:
        """Binary foreground mask (importance ≥ threshold)."""
        imp = self.build_importance_map(
            ct_slices, filenames, patch_rows, patch_cols, patch_size
        )
        return (imp >= threshold).float()

    # ------------------------------------------------------------------
    # ROI PSNR metrics
    # ------------------------------------------------------------------
    def _masked_psnr(self, recon, original, mask):
        """Compute PSNR inside a binary mask."""
        psnr_vals = []
        for b in range(recon.shape[0]):
            m = mask[b, 0].bool()
            if m.sum() < 10:
                continue
            r = recon[b][:, m]
            o = original[b][:, m]
            mse = F.mse_loss(r, o).item()
            psnr_vals.append(10 * np.log10(1.0 / (mse + 1e-8)))
        return np.mean(psnr_vals) if psnr_vals else 0.0

    def compute_roi_metrics(self, reconstructed, original, filenames,
                            patch_rows=None, patch_cols=None):
        """Compute ROI, background, lung, and nodule PSNR."""
        recon = reconstructed.detach()
        orig  = original.detach()

        imp_map     = self.build_importance_map(orig, filenames, patch_rows, patch_cols)
        fg_mask     = (imp_map >= 0.25).float()
        bg_mask     = (imp_map <  0.25).float()
        lung_mask   = (imp_map >= 0.75).float()
        nodule_mask = (imp_map >= 0.95).float()

        return {
            'roi_psnr':    self._masked_psnr(recon, orig, fg_mask),
            'bg_psnr':     self._masked_psnr(recon, orig, bg_mask),
            'global_psnr': 10 * np.log10(1.0 / (F.mse_loss(recon, orig).item() + 1e-8)),
            'roi_psnr_gain': (self._masked_psnr(recon, orig, fg_mask)
                            - self._masked_psnr(recon, orig, bg_mask)),
            'lung_psnr':   self._masked_psnr(recon, orig, lung_mask),
            'nodule_psnr': self._masked_psnr(recon, orig, nodule_mask),
            'fg_fraction': fg_mask.mean().item(),
        }

    # ------------------------------------------------------------------
    # Organ map quality metrics
    # ------------------------------------------------------------------
    def compute_organ_map_metrics(self, organ_maps, original, filenames,
                                  patch_rows=None, patch_cols=None):
        """Compute sparsity, foreground alignment, diversity, entropy of organ maps."""
        maps = organ_maps.detach()
        B, K, H, W = maps.shape

        imp_map   = self.build_importance_map(original, filenames, patch_rows, patch_cols)
        fg_weight = imp_map.squeeze(1)

        sparsity_vals, alignment_vals, diversity_vals, entropy_vals = [], [], [], []

        for b in range(B):
            fg = fg_weight[b]
            for k in range(K):
                m = maps[b, k]
                sparsity_vals.append((m < 0.1).float().mean().item())
                total_mass = m.sum().item() + 1e-8
                alignment_vals.append((m * fg).sum().item() / total_mass)
                p = m / (m.sum() + 1e-8)
                ent = -(p * (p + 1e-8).log()).sum().item()
                entropy_vals.append(ent / np.log(H * W))

            if K > 1:
                flat = maps[b].reshape(K, -1)
                flat_norm = F.normalize(flat, dim=1)
                sim_mat = torch.mm(flat_norm, flat_norm.t())
                off_diag = ~torch.eye(K, dtype=torch.bool, device=maps.device)
                diversity_vals.append(1.0 - sim_mat[off_diag].mean().item())

        return {
            'map_sparsity':        np.mean(sparsity_vals),
            'map_fg_alignment':    np.mean(alignment_vals),
            'map_inter_diversity': np.mean(diversity_vals) if diversity_vals else 0.0,
            'map_entropy':         np.mean(entropy_vals),
        }

    # ------------------------------------------------------------------
    # Spatial bitrate map (static)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_spatial_bitrate_map(indices, latent_hw, codebook_size,
                                    image_hw=(256, 256),
                                    precomputed_bits=None):
        """Create a per‑pixel bitrate map from codebook indices.

        Args:
            indices: (B, n_codebooks, N) hard indices.
            latent_hw: (H_lat, W_lat) spatial shape of the latent.
            codebook_size: Number of codebook entries.
            image_hw: (H, W) of the output map.
            precomputed_bits: Optional precomputed per‑code entropy values.

        Returns:
            Bitrate map (B, H, W) with values normalised to [0,1] per image.
        """
        B, n_cb, N = indices.shape
        H_lat, W_lat = latent_hw
        device = indices.device

        idx_spatial = indices.view(B, n_cb, H_lat, W_lat)
        entropy_map = torch.zeros(B, H_lat, W_lat, device=device, dtype=torch.float)

        for cb in range(n_cb):
            idx_cb = idx_spatial[:, cb, :, :]
            oh = F.one_hot(idx_cb, codebook_size).float()
            if precomputed_bits is not None:
                bits = precomputed_bits[cb].to(device)
            else:
                global_counts = oh.sum(dim=(0, 1, 2))
                global_probs = global_counts / (global_counts.sum() + 1e-10)
                bits = -torch.log2(global_probs + 1e-10)
            entropy_map += (oh * bits.view(1, 1, 1, -1)).sum(dim=-1)

        # Normalise per image
        for b in range(B):
            m = entropy_map[b]
            entropy_map[b] = (m - m.min()) / (m.max() - m.min() + 1e-8)

        return F.interpolate(
            entropy_map.unsqueeze(1), size=image_hw,
            mode='bilinear', align_corners=False,
        ).squeeze(1)

    # ------------------------------------------------------------------
    # Bitrate–organ alignment
    # ------------------------------------------------------------------
    def bitrate_organ_alignment(self, bitrate_map, organ_maps,
                                threshold=None):
        """Correlation between bitrate and organ map activation."""
        B, K, H, W = organ_maps.shape
        max_organ = organ_maps.max(dim=1).values
        fg_mass_vals, corr_vals = [], []

        for b in range(B):
            bm = bitrate_map[b]
            om = max_organ[b]

            if threshold is not None:
                organ_mask = (om > threshold).float()
                if organ_mask.sum() > 10:
                    fg_mass_vals.append(
                        ((bm * organ_mask).sum() / (bm.sum() + 1e-8)).item()
                    )
            else:
                fg_mass_vals.append(
                    ((bm * om).sum() / (bm.sum() + 1e-8)).item()
                )

            bm_c = bm.reshape(-1) - bm.mean()
            om_c = om.reshape(-1) - om.mean()
            corr = (bm_c * om_c).sum() / (bm_c.norm() * om_c.norm() + 1e-8)
            corr_vals.append(corr.item())

        return {
            'bitrate_fg_mass':    np.mean(fg_mass_vals) if fg_mass_vals else 0.0,
            'bitrate_organ_corr': np.mean(corr_vals),
        }

    # ------------------------------------------------------------------
    # Reconstruction error alignment
    # ------------------------------------------------------------------
    def error_organ_alignment(self, original, reconstructed, organ_maps,
                              filenames, patch_rows=None, patch_cols=None,
                              threshold=0.5):
        """Ratio of reconstruction error inside vs outside organ regions."""
        diff = (original - reconstructed).abs().mean(dim=1)
        imp_map = self.build_importance_map(
            original, filenames, patch_rows, patch_cols
        ).squeeze(1)

        max_organ = organ_maps.max(dim=1).values
        roi_mask = ((max_organ > threshold) | (imp_map >= 0.75)).float()

        ratio_vals = []
        for b in range(diff.shape[0]):
            d = diff[b]
            om = roi_mask[b]
            bg = 1.0 - om
            if om.sum() > 10 and bg.sum() > 10:
                roi_err = (d * om).sum() / (om.sum() + 1e-8)
                bg_err = (d * bg).sum() / (bg.sum() + 1e-8)
                ratio_vals.append((roi_err / (bg_err + 1e-8)).item())

        return {'roi_error_ratio': np.mean(ratio_vals) if ratio_vals else 1.0}

    # ------------------------------------------------------------------
    # Full evaluation pass
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_full_evaluation(self, model, val_loader, epoch=0, label='model',
                            n_paper_figures=8, save_figures=True,
                            cache_outputs=False):
        """Evaluate all metrics over a validation loader.

        Args:
            model: MedicalCompressor model.
            val_loader: DataLoader for validation.
            epoch: Epoch number for figure naming.
            label: Label string for logs.
            n_paper_figures: Maximum number of paper figures to save.
            save_figures: Whether to save paper figures.
            cache_outputs: If True, cache all outputs for faster repeated analysis.

        Returns:
            Dictionary of averaged metrics.
        """
        model.eval()
        device = self.device
        codebook_size = model.codebook_size
        n_codebooks = model.n_codebooks

        # First pass: accumulate global codebook counts
        global_counts = torch.zeros(n_codebooks, codebook_size, device='cpu')
        cached = [] if cache_outputs else None

        for batch in val_loader:
            images = batch['image'].to(device)
            output = model(images)
            indices = output['indices'].cpu()

            for cb in range(n_codebooks):
                global_counts[cb] += torch.bincount(
                    indices[:, cb, :].reshape(-1), minlength=codebook_size
                )

            if cache_outputs:
                cached.append({
                    'images':        images.cpu(),
                    'reconstructed': output['reconstructed'].cpu(),
                    'organ_maps':    output['organ_maps'].cpu(),
                    'indices':       indices,
                    'filenames':     batch['filename'],
                    'patch_rows':    batch['patch_row'].tolist(),
                    'patch_cols':    batch['patch_col'].tolist(),
                })

        global_probs = global_counts / (global_counts.sum(dim=1, keepdim=True) + 1e-10)
        bits_per_code = -torch.log2(global_probs + 1e-10).to(device)

        # Second pass: compute per‑batch metrics
        all_metrics = defaultdict(list)
        n_figs_saved = 0
        data_iter = cached if cache_outputs else val_loader

        for batch_idx, batch in enumerate(data_iter):
            if cache_outputs:
                images     = batch['images'].to(device)
                recon      = batch['reconstructed'].to(device)
                organ_maps = batch['organ_maps'].to(device)
                indices    = batch['indices'].to(device)
                filenames  = batch['filenames']
                patch_rows = batch['patch_rows']
                patch_cols = batch['patch_cols']
            else:
                images = batch['image'].to(device)
                filenames = batch['filename']
                patch_rows = batch['patch_row'].tolist()
                patch_cols = batch['patch_col'].tolist()
                output = model(images)
                recon = output['reconstructed']
                organ_maps = output['organ_maps']
                indices = output['indices']

            brt_map = self.compute_spatial_bitrate_map(
                indices, self.latent_hw, codebook_size,
                self.image_hw, precomputed_bits=bits_per_code,
            )

            roi   = self.compute_roi_metrics(recon, images, filenames, patch_rows, patch_cols)
            omap  = self.compute_organ_map_metrics(organ_maps, images, filenames, patch_rows, patch_cols)
            align = self.bitrate_organ_alignment(brt_map, organ_maps)
            err   = self.error_organ_alignment(images, recon, organ_maps, filenames, patch_rows, patch_cols)

            for d in (roi, omap, align, err):
                for k, v in d.items():
                    all_metrics[k].append(v)

            if save_figures and n_figs_saved < n_paper_figures:
                n_save = min(images.shape[0], n_paper_figures - n_figs_saved)
                self.save_paper_figure(
                    images, recon, organ_maps, brt_map,
                    filenames=filenames, patch_rows=patch_rows, patch_cols=patch_cols,
                    epoch=epoch, save_subdir=label,
                    batch_idx=batch_idx, n_images=n_save,
                )
                n_figs_saved += n_save

        results = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        print(f'\n[{label}] Diagnostic metrics:')
        for k, v in results.items():
            print(f'  {k:<30} {v:.4f}')
        return results

    # ------------------------------------------------------------------
    # Paper figure (5‑panel)
    # ------------------------------------------------------------------
    def save_paper_figure(self, original, reconstructed, organ_maps, bitrate_map,
                          filenames, patch_rows, patch_cols,
                          epoch, save_subdir='', batch_idx=0, n_images=4):
        """Save a 5‑panel diagnostic figure: original, importance map, organ map,
        spatial bitrate, reconstruction error."""
        save_dir = f'{self.save_dir}/paper_figures/{save_subdir}'
        os.makedirs(save_dir, exist_ok=True)

        B = min(original.shape[0], n_images)
        max_organ = organ_maps.max(dim=1).values
        imp_maps = self.build_importance_map(
            original, filenames, patch_rows, patch_cols
        )

        for b in range(B):
            orig_np  = original[b].cpu().permute(1, 2, 0).numpy()
            recon_np = reconstructed[b].cpu().permute(1, 2, 0).numpy()
            org_np   = max_organ[b].cpu().numpy()
            brt_np   = bitrate_map[b].cpu().numpy()
            imp_np   = imp_maps[b, 0].cpu().numpy()
            err_np   = np.abs(orig_np - recon_np).mean(-1)

            mse  = np.mean((orig_np - recon_np) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-8))

            org_c = org_np.reshape(-1) - org_np.mean()
            brt_c = brt_np.reshape(-1) - brt_np.mean()
            corr  = float(np.dot(org_c, brt_c) /
                          (np.linalg.norm(org_c) * np.linalg.norm(brt_c) + 1e-8))

            roi_mask = ((org_np > 0.5) | (imp_np >= 0.75))
            if roi_mask.sum() > 10 and (~roi_mask).sum() > 10:
                ratio = err_np[roi_mask].mean() / (err_np[~roi_mask].mean() + 1e-8)
            else:
                ratio = 1.0

            # Grayscale or RGB display
            def to_display(img_rgb):
                if np.allclose(img_rgb[:, :, 0], img_rgb[:, :, 1], atol=0.02):
                    return img_rgb[:, :, 0], 'gray'
                return img_rgb, None

            img_disp, cmap = to_display(np.clip(orig_np, 0, 1))
            row_i = int(patch_rows[b]) if patch_rows is not None else 0
            col_i = int(patch_cols[b]) if patch_cols is not None else 0

            fig, axes = plt.subplots(1, 5, figsize=(25, 5))

            axes[0].imshow(img_disp, cmap=cmap)
            axes[0].set_title(f'Original\n(patch row={row_i}, col={col_i})',
                              fontsize=11, fontweight='bold')

            axes[1].imshow(img_disp, cmap='gray')
            im1 = axes[1].imshow(imp_np, cmap='RdYlGn', alpha=0.6, vmin=0, vmax=1)
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            axes[1].set_title('Importance Map\n0=air | 0.3=tissue | 0.6=medi\n0.8=lung | 1.0=nodule', fontsize=9)

            axes[2].imshow(img_disp, cmap='gray')
            im2 = axes[2].imshow(org_np, cmap='hot', alpha=0.55, vmin=0, vmax=1)
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            axes[2].set_title('Organ Map', fontsize=11)

            axes[3].imshow(img_disp, cmap='gray')
            im3 = axes[3].imshow(brt_np, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
            plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
            axes[3].set_title(f'Spatial Bitrate\n(r={corr:.2f})', fontsize=11)

            axes[4].imshow(img_disp, cmap='gray')
            im4 = axes[4].imshow(err_np, cmap='RdYlGn_r', alpha=0.6,
                                 vmin=0, vmax=np.percentile(err_np, 95))
            plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
            verdict = '✓ preserved' if ratio < 1.0 else '✗ not preserved'
            axes[4].set_title(f'Error (ROI/BG={ratio:.2f}, {verdict})', fontsize=11)

            for ax in axes:
                ax.axis('off')

            plt.suptitle(f'Epoch {epoch} | PSNR: {psnr:.2f} dB | '
                         f'Bitrate-Organ Corr: {corr:.2f} | {filenames[b]}',
                         fontsize=10)
            plt.tight_layout()
            base = f'{save_dir}/epoch_{epoch:03d}_b{batch_idx}_s{b}'
            plt.savefig(base + '.pdf', dpi=200, bbox_inches='tight')
            plt.savefig(base + '.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved: {base}.png')

    # ------------------------------------------------------------------
    # Ablation tools
    # ------------------------------------------------------------------
    @torch.no_grad()
    def three_condition_ablation(self, model_full, model_frozen, model_baseline,
                                 val_loader, epoch):
        """Run evaluation on three models: full, frozen attention, no attention."""
        print('Running 3-condition ablation...')
        results = {}
        for label, model in [
            ('A_no_attention',     model_baseline),
            ('B_frozen_attention', model_frozen),
            ('C_full_model',       model_full),
        ]:
            results[label] = self.run_full_evaluation(
                model, val_loader, epoch=epoch, label=label,
                save_figures=(label == 'C_full_model'),
            )
        return results

    def print_ablation_table(self, results):
        """Print a formatted table of ablation metrics."""
        conditions = ['A_no_attention', 'B_frozen_attention', 'C_full_model']
        metrics = [
            ('bitrate_organ_corr', 'Bitrate–Organ Correlation (↑)'),
            ('bitrate_fg_mass',    'Bitrate Mass on Anatomy (↑)'),
            ('roi_error_ratio',    'ROI/BG Error Ratio (↓)'),
            ('roi_psnr_gain',      'ROI PSNR Gain (↑)'),
            ('lung_psnr',          'Lung PSNR (↑)'),
            ('nodule_psnr',        'Nodule PSNR (↑)'),
            ('map_fg_alignment',   'Organ Map FG Alignment (↑)'),
        ]
        print(f'\n{"="*75}')
        print('THREE-CONDITION ABLATION — Summary')
        print(f'{"="*75}')
        print(f'{"Metric":<38} {"A":>9} {"B":>9} {"C":>9}')
        print(f'{"-"*75}')
        for key, label in metrics:
            vals = [results.get(c, {}).get(key, float('nan')) for c in conditions]
            print(f'{label:<38}' + ''.join(f' {v:>9.3f}' for v in vals))
        print(f'{"="*75}')

    def plot_ablation_bars(self, results, epoch=150):
        """Plot a bar chart comparing the three conditions."""
        conditions = ['A_no_attention', 'B_frozen_attention', 'C_full_model']
        x_labels   = ['No Attention', 'Frozen\nAttention', 'Full Model\n(Ours)']
        colors     = ['#d62728', '#ff7f0e', '#2ca02c']
        metrics    = [
            ('bitrate_organ_corr', 'Bitrate–Organ\nCorrelation (↑)', False),
            ('bitrate_fg_mass',    'Bitrate Mass\non Anatomy (↑)',   False),
            ('roi_error_ratio',    'ROI/BG Error\nRatio (↓)',        True),
            ('lung_psnr',          'Lung PSNR (↑)',                  False),
            ('nodule_psnr',        'Nodule PSNR (↑)',                False),
        ]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        for ax, (key, ylabel, lower_better) in zip(axes, metrics):
            vals = [results.get(c, {}).get(key, 0) for c in conditions]
            bars = ax.bar(x_labels, vals, color=colors, width=0.5,
                          edgecolor='black', linewidth=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            ax.set_title(ylabel, fontsize=11, fontweight='bold')
            ax.set_ylim(0, max(vals) * 1.25 + 0.01)
            ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            best_idx = int(np.argmin(vals)) if lower_better else int(np.argmax(vals))
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(2.5)

        plt.suptitle(f'Three-Condition Ablation — Epoch {epoch}', fontsize=12)
        plt.tight_layout()
        base = f'{self.save_dir}/ablation_bars_epoch_{epoch}'
        plt.savefig(base + '.pdf', dpi=200, bbox_inches='tight')
        plt.savefig(base + '.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Ablation bar chart saved: {base}.png')
