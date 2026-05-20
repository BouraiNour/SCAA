"""
Preserving Diagnosis, Reducing Bits: Sparsity‑Controlled Linear Anatomical
Attention for Medical Image Compression.
"""

from .dataset import MedicalImageDataset, build_loaders, get_reproducible_split
from .model import (
    MedicalCompressor,
    AnatomicalAttention,
    SparsityPriorGenerator,
    AnatomicalEncoder,
    AnatomicalDecoder,
    ResidualVectorQuantizer,
    ResidualBlock,
    AttentionBlock,
    OrgMapModulation,
)
from .loss import MedicalCompressionLoss
from .entropy import DiscreteEntropyEstimator
from .evaluator import DiagnosticEvaluator
from .trainer import MedicalCompressionTrainer
from .utils import IdentityAttention
