"""
Standalone evaluation script for the compression model.

Usage:
    python evaluate.py --checkpoint path/to/best_model.pth --config configs/default_config.json
"""

import argparse
import json
import torch
from src.model.compressor import MedicalCompressor
from src.dataset import build_loaders
from src.evaluator import DiagnosticEvaluator


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained anatomical‑attention compression model.'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint.')
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                        help='Path to the configuration JSON file.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory from config.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation figures and metrics.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.data_dir:
        config['data_dir'] = args.data_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build validation loader
    _, val_loader = build_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        modality=config.get('modality', 'ct'),
        patch_size=config.get('patch_size', 256),
        val_frac=config.get('val_frac', 0.15),
        num_workers=config.get('num_workers', 4),
        seed=42,
        split_save_path=config.get('split_save_path'),
        metadata_csv=config.get('metadata_csv'),
    )

    # Load model
    model = MedicalCompressor(
        in_channels=config['in_channels'],
        num_organs=config['num_organs'],
        latent_dim=config.get('latent_dim', 256),
        n_codebooks=config.get('n_codebooks', 8),
        codebook_size=config.get('codebook_size', 1024),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Evaluate
    evaluator = DiagnosticEvaluator(
        masks_dir=config.get('masks_dir', ''),
        device=str(device),
        save_dir=args.output_dir,
        latent_hw=config.get('latent_hw', (16, 16)),
        image_hw=config.get('image_size', (256, 256)),
        patch_size=config.get('patch_size', 256),
    )

    results = evaluator.run_full_evaluation(
        model, val_loader,
        epoch=checkpoint['epoch'],
        label='Standalone_Eval',
        save_figures=True,
    )

    with open(f'{args.output_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete. Results saved to", args.output_dir)


if __name__ == '__main__':
    main()
