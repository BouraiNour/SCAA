"""
Entry point for training the medical image compression model.

Usage:
    python train.py --config configs/default_config.json [--resume checkpoint.pth]
"""

import argparse
import json
from src.trainer import MedicalCompressionTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Train the anatomical‑attention compression model.'
    )
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                        help='Path to configuration JSON file.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint to resume training.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    trainer = MedicalCompressionTrainer(config, output_dir=config.get('output_dir', 'training_output'))
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
