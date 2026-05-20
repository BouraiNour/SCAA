# Preserving Diagnosis, Reducing Bits: Sparsity‑Controlled Linear Anatomical Attention for Medical Image Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20314843.svg)](https://doi.org/10.5281/zenodo.20314843)

Official PyTorch implementation of the paper **"Preserving Diagnosis, Reducing Bits: Sparsity‑Controlled Linear Anatomical Attention for Medical Image Compression"**.

Authors: [Your Name], [Co‑authors]

---

## Abstract

The compression of medical images demands both aggressive bitrate reduction and the faithful reconstruction of diagnostically critical structures. Standard codecs allocate bits uniformly, ignoring clinical salience, while region‑of‑interest methods require external segmentation annotations. We introduce the \textbf{Sparsity-Controlled Anatomical Attention (SCAA)} framework: a self‑supervised mechanism that learns to prioritise anatomically significant regions during compression without manual labels. SCAA couples a \emph{SparsityPriorGenerator} (producing soft organ‑attention maps via a learnable temperature) with a linear‑complexity \emph{AnatomicalAttention} block that conditions latent feature routing, achieving $\mathcal{O}(Ld^{2})$ complexity – about three orders of magnitude lower than standard self‑attention. 

Evaluated on a dedicated 2D CT dataset derived from LUNA16 ($6\,216$ axial slices, patient‑level split), our full compression pipeline attains \textbf{29.84 dB PSNR at 0.310 bpp} (SSIM 0.9695, LPIPS 0.052). The spatial bitrate–organ correlation reaches 0.188, confirming that bits are preferentially allocated to anatomically attended regions. These results establish SCAA as a computationally efficient, annotation‑free alternative to supervision‑dependent ROI coding for medical image compression.

---

## Dataset: LUNA16‑DP2D

We release the **LUNA16‑DiagnosticPreservation‑2D** dataset, derived from the LUNA16 challenge:

- **6,216 CT slices** (512×512, int16 HU)
- **6,216 lung masks** + **6,216 nodule masks**
- **metadata.csv** with per‑slice statistics

| Source | Link |
|--------|------|
| HuggingFace | [nourbourai/luna16-dp2d](https://huggingface.co/datasets/nourbourai/luna16-dp2d) |
| Zenodo (archival) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20314843.svg)](https://doi.org/10.5281/zenodo.20314843) |

*Dataset is CC BY 4.0.*

---

## Method Overview

### Sparsity‑Controlled Linear Anatomical Attention
1. **Sparsity Prior Generator**: produces K soft organ masks via a learnable temperature‑scaled sigmoid. High temperature → near‑binary, sparse masks.
2. **Linear Attention (ELU kernel)**: O(N) complexity, applied only to gated features (anatomical regions), directing capacity where it matters.
3. **Multi‑Scale Priors**: fine (organ boundary) and coarse (whole‑organ) priors fused with a learnable weight.

### Loss Design
- **ROI‑weighted MSE** (from a differentiable foreground mask) penalises errors on anatomy more heavily.
- **Foreground alignment** and **diversity** losses ensure masks match the body and remain distinct.
- **Rate–distortion–perceptual** trade‑off with warm‑up schedules.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA recommended

```bash
pip install -r requirements.txt
```
# Quick Start

## 1. Clone & install
git clone https://github.com/YourUsername/luna16-diagnostic-compression.git
cd luna16-diagnostic-compression
pip install -r requirements.txt

## 2. Download dataset
huggingface-cli download nourbourai/luna16-dp2d --repo-type dataset --local-dir data/

## 3. Train
python train.py --config configs/default_config.json

## 4. Evaluate
python evaluate.py --checkpoint training_output/checkpoints/best_model.pth --data_dir data/

# Repository Structure

SCAA/
├── configs/
│   └── default_config.json
├── data/                  
├── src/
│   ├── dataset.py
│   ├── model/
│   │   ├── blocks.py
│   │   ├── anatomical_attention.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── quantizer.py
│   │   └── compressor.py
│   ├── loss.py
│   ├── entropy.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── utils.py
├── train.py
├── evaluate.py
├── requirements.txt
├── LICENSE
└── README.md

# Citation

@article{yourname2026preserving,
  title={Preserving Diagnosis, Reducing Bits: Sparsity-Controlled Linear Anatomical Attention for Medical Image Compression},
   author = {Nour El Houda Bourai,  Hayet Farida Merouani,  Akila Djebbar},
  journal={Signal, Image and Video Processing},
  year={2026}
}

@dataset{nourbourai_luna16-dp2d,
  author       = {Nour El Houda Bourai},
  title        = {LUNA16-DP2D: 2D CT Slices with Anatomical Masks for Diagnostic-Preservation Image Compression},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20314843},
  url          = {https://doi.org/10.5281/zenodo.20314843}
}

# Acknowledgments
We thank the LUNA16 challenge organisers for the original CT data.

#  Contact

For questions, open an issue.


