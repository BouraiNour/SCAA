"""
Medical Image Dataset and DataLoader builders.

Provides:
- MedicalImageDataset: on‑the‑fly patch extraction, modality‑aware normalisation,
  pre‑patched mode, error‑resilient loading.
- get_reproducible_split: deterministic train/val file split.
- build_loaders: convenience wrapper that creates DataLoaders with optional
  patient‑level splitting using a metadata CSV.
"""

import os
import glob
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Optional, Dict
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize(arr: np.ndarray, modality: str) -> np.ndarray:
    """Normalise a 2D medical image based on modality.

    Args:
        arr: Input array (H, W).
        modality: One of 'ct', 'mri', 'xray', or 'auto'. Case‑insensitive.

    Returns:
        Float32 array clipped to [0, 1].
    """
    arr = arr.astype(np.float32)
    mod = modality.lower().strip()

    if mod in ('ct', 'ct_chest', 'ct_abdomen'):
        CT_MIN, CT_MAX = -1000.0, 400.0
        arr = np.clip(arr, CT_MIN, CT_MAX)
        arr = (arr - CT_MIN) / (CT_MAX - CT_MIN)

    elif mod in ('mri', 'mri_brain'):
        p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
        if p99 > p1:
            arr = np.clip(arr, p1, p99)
            arr = (arr - p1) / (p99 - p1)
        else:
            arr = np.zeros_like(arr)

    elif mod == 'xray':
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)

    else:   # 'auto' or unknown
        lo, hi = arr.min(), arr.max()
        if hi - lo > 1e-8:
            arr = (arr - lo) / (hi - lo + 1e-8)
        else:
            arr = np.zeros_like(arr)

    return arr.clip(0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Patch extraction (legacy, now integrated inside the dataset)
# ---------------------------------------------------------------------------

def extract_patches(arr: np.ndarray, patch_size: int = 256) -> List[np.ndarray]:
    """Divide a 2D array into square patches.

    Args:
        arr: Input array (H, W).
        patch_size: Edge length of each patch.

    Returns:
        List of patch arrays.
    """
    H, W = arr.shape
    target_H = max(patch_size, round(H / patch_size) * patch_size)
    target_W = max(patch_size, round(W / patch_size) * patch_size)

    if (H, W) != (target_H, target_W):
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        t = F.interpolate(t, size=(target_H, target_W),
                          mode='bilinear', align_corners=False)
        arr = t.squeeze().numpy()

    patches = []
    for r in range(target_H // patch_size):
        for c in range(target_W // patch_size):
            patches.append(arr[r * patch_size:(r + 1) * patch_size,
                               c * patch_size:(c + 1) * patch_size])
    return patches


def to_tensor(arr2d: np.ndarray) -> torch.Tensor:
    """Convert a 2D array to a 3‑channel tensor (replicates the channel).

    Args:
        arr2d: (H, W) array.

    Returns:
        Tensor of shape (3, H, W).
    """
    t = torch.from_numpy(arr2d).unsqueeze(0).float()
    return t.repeat(3, 1, 1)


# ---------------------------------------------------------------------------
# Medical Image Dataset
# ---------------------------------------------------------------------------

class MedicalImageDataset(Dataset):
    """Medical image dataset with on‑the‑fly patching and modality normalisation.

    Every item is a dictionary containing:
        - image: (3, patch_size, patch_size) tensor
        - filename: original .npy slice name (for mask loading during evaluation)
        - original_min / original_max / original_shape: raw HU statistics
        - patch_row / patch_col: grid indices of the patch
        - modality: string

    Two modes:
        - On‑the‑fly (default): builds patches from whole‑slice .npy files.
        - Pre‑patched: each file is already a normalised patch (prepatched=True).

    Args:
        data_dir: Directory containing .npy or .dcm files (searched recursively).
        modality: Modality string for normalisation.
        patch_size: Edge length of square patches. If None, whole slice is used.
        target_size: Desired output size after optional interpolation.
        transform: Optional torchvision transform applied to the image tensor.
        prepatched: If True, each file is a patch (no further patching).
        patch_sep: Separator used to identify original filename from patch name.
        patch_map_path: JSON mapping patch filenames to original slice names.
        verbose: Print dataset statistics at init.
    """

    def __init__(self,
                 data_dir:       str,
                 modality:       str  = 'auto',
                 patch_size:     int  = 256,
                 target_size:    Tuple = (256, 256),
                 transform            = None,
                 prepatched:     bool = False,
                 patch_sep:      str  = '_patch',
                 patch_map_path: Optional[str] = None,
                 verbose:        bool = True):

        self.modality       = modality
        self.patch_size     = patch_size
        self.target_size    = target_size
        self.transform      = transform
        self.prepatched     = prepatched
        self.patch_sep      = patch_sep
        self.patch_map_path = patch_map_path

        self.patch_to_original = None
        if patch_map_path is not None:
            with open(patch_map_path, 'r') as f:
                self.patch_to_original = json.load(f)
            if verbose:
                print(f"  Loaded patch mapping from {patch_map_path}")

        # Discover files
        raw_paths = sorted(
            glob.glob(os.path.join(data_dir, '**', '*.npy'), recursive=True) +
            glob.glob(os.path.join(data_dir, '**', '*.dcm'), recursive=True)
        )
        if not raw_paths:
            raise FileNotFoundError(f"No .npy or .dcm files found in {data_dir}")

        # Filter corrupt files
        valid_paths, n_corrupt = [], 0
        for p in raw_paths:
            try:
                arr = np.load(p, mmap_mode='r')
                if arr.size > 0:
                    valid_paths.append(p)
                else:
                    n_corrupt += 1
            except Exception:
                n_corrupt += 1

        if n_corrupt > 0:
            print(f"  ⚠ {n_corrupt} corrupt/empty files skipped at init.")

        if prepatched:
            self.index = valid_paths
            self.original_to_patches     = {}
            self.patch_to_original_built = {}
            for i, path in enumerate(self.index):
                fname = os.path.basename(path)
                if self.patch_to_original is not None:
                    base = self.patch_to_original.get(fname)
                    if base is None:
                        raise KeyError(f"Patch file {fname} not found in mapping")
                else:
                    base = (fname.split(self.patch_sep)[0] + '.npy'
                            if self.patch_sep in fname else fname)
                self.original_to_patches.setdefault(base, []).append(i)
                self.patch_to_original_built[i] = base

            if verbose:
                print(f"  MedicalImageDataset (prepatched) | "
                      f"{len(self.index)} patches, "
                      f"{len(self.original_to_patches)} original files")
        else:
            self.index: List[Tuple[str, int, int]] = []
            for path in valid_paths:
                try:
                    arr  = np.load(path, mmap_mode='r')
                    arr2 = self._to_2d(arr)
                    H, W = arr2.shape
                    if patch_size is not None:
                        tH = max(patch_size, round(H / patch_size) * patch_size)
                        tW = max(patch_size, round(W / patch_size) * patch_size)
                        for r in range(tH // patch_size):
                            for c in range(tW // patch_size):
                                self.index.append((path, r, c))
                    else:
                        self.index.append((path, 0, 0))
                except Exception:
                    pass

            if verbose:
                n_files   = len(valid_paths)
                n_patches = len(self.index)
                print(f"  MedicalImageDataset | modality={modality} | "
                      f"patch_size={patch_size}")
                print(f"  {n_files} files → {n_patches} patches "
                      f"({n_patches/max(n_files,1):.1f}× expansion)")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_2d(arr: np.ndarray) -> np.ndarray:
        """Squeeze array to 2D, handling common medical formats."""
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3):
                return arr[0]
            return arr[arr.shape[0] // 2]
        return arr.reshape(arr.shape[-2], arr.shape[-1])

    # -------------------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        """Return a patch with metadata, with fallback retries on error."""
        max_attempts = 10
        current_idx  = idx

        for attempt in range(max_attempts):

            if self.prepatched:
                # ── PRE-PATCHED MODE ─────────────────────────────────────
                path = self.index[current_idx]
                try:
                    arr   = np.load(path).astype(np.float32)
                    image = to_tensor(arr)
                    if self.transform:
                        image = self.transform(image)

                    original_base = self.patch_to_original_built[current_idx]

                    return {
                        'image':          image,
                        'filename':       original_base,          # for mask loading
                        'original_min':   0.0,
                        'original_max':   1.0,
                        'original_shape': arr.shape,
                        'file_path':      path,
                        'original_file':  original_base,
                        'patch_row':      0,
                        'patch_col':      0,
                        'modality':       'prepatched',
                    }
                except Exception as e:
                    print(f"Warning: prepatched fallback for {path} ({e})")

            else:
                # ── ON-THE-FLY MODE ───────────────────────────────────────
                path, row, col = self.index[current_idx]
                try:
                    raw       = np.load(path, mmap_mode='c')
                    arr       = self._to_2d(raw).astype(np.float32)
                    raw_min   = float(arr.min())
                    raw_max   = float(arr.max())
                    raw_shape = arr.shape

                    arr = normalize(arr, self.modality)

                    if self.patch_size is not None:
                        H, W = arr.shape
                        tH   = max(self.patch_size,
                                   round(H / self.patch_size) * self.patch_size)
                        tW   = max(self.patch_size,
                                   round(W / self.patch_size) * self.patch_size)
                        if (H, W) != (tH, tW):
                            t   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                            t   = F.interpolate(t, size=(tH, tW),
                                                mode='bilinear', align_corners=False)
                            arr = t.squeeze().numpy()
                        r0, r1 = row * self.patch_size, (row + 1) * self.patch_size
                        c0, c1 = col * self.patch_size, (col + 1) * self.patch_size
                        arr = arr[r0:r1, c0:c1]
                    else:
                        if arr.shape != self.target_size:
                            t   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                            t   = F.interpolate(t, size=self.target_size,
                                                mode='bilinear', align_corners=False)
                            arr = t.squeeze().numpy()

                    image = to_tensor(arr)
                    if self.transform:
                        image = self.transform(image)

                    return {
                        'image':          image,
                        'filename':       os.path.basename(path),  # for mask loading
                        'original_min':   raw_min,
                        'original_max':   raw_max,
                        'original_shape': raw_shape,
                        'file_path':      path,
                        'patch_row':      row,
                        'patch_col':      col,
                        'modality':       self.modality,
                    }
                except Exception as e:
                    print(f"Warning: on-the-fly fallback for {path} ({e})")

            current_idx = (current_idx + 1) % len(self.index)

        raise RuntimeError(
            f"Failed to load a valid sample after {max_attempts} attempts. "
            f"Last attempted index: {current_idx} "
            f"(file: {self.index[current_idx]}). "
            f"Please check data integrity."
        )


# ---------------------------------------------------------------------------
# Reproducible split helper
# ---------------------------------------------------------------------------

def get_reproducible_split(file_paths:   List[str],
                           train_ratio:  float = 0.8,
                           seed:         int   = 42,
                           save_path:    Optional[str] = None
                           ) -> Tuple[List[str], List[str]]:
    """Deterministic file‑level train/val split.

    Args:
        file_paths: List of file paths to split.
        train_ratio: Fraction for training.
        seed: Random seed.
        save_path: If given, save split to JSON; if it already exists, load it.

    Returns:
        (train_paths, val_paths)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if save_path is not None and Path(save_path).exists():
        print(f"♻️ Loading existing split from {save_path}")
        with open(save_path, 'r') as f:
            split = json.load(f)
        return split['train'], split['val']

    print(f"🆕 Creating new reproducible split (seed: {seed})")
    shuffled_paths = sorted(file_paths)
    random.shuffle(shuffled_paths)
    split_idx    = int(len(shuffled_paths) * train_ratio)
    train_paths  = shuffled_paths[:split_idx]
    val_paths    = shuffled_paths[split_idx:]

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({'train': train_paths, 'val': val_paths}, f)

    return train_paths, val_paths


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_loaders(data_dir:        Optional[str] = None,
                  batch_size:      int   = 4,
                  modality:        str   = 'ct',
                  patch_size:      int   = 256,
                  val_frac:        float = 0.20,
                  test_frac:       float = 0.0,
                  num_workers:     int   = 4,
                  seed:            int   = 42,
                  prepatched_dir:  Optional[str] = None,
                  split_save_path: Optional[str] = None,
                  metadata_csv:    Optional[str] = None) -> Tuple[DataLoader, ...]:
    """Build train/val (and optionally test) DataLoaders.

    Supports both file‑level and patient‑level (via metadata_csv) splitting.
    Patient‑level uses the 'original_scan' column to group slices.

    Args:
        data_dir: Directory of raw slice .npy files (on‑the‑fly mode).
        batch_size: Batch size.
        modality: Modality for normalisation.
        patch_size: Patch edge length.
        val_frac: Fraction for validation.
        test_frac: Fraction for test (only with patient split).
        num_workers: DataLoader workers.
        seed: Random seed for split.
        prepatched_dir: Directory of pre‑patched files.
        split_save_path: Path to save/load the split JSON.
        metadata_csv: Path to metadata.csv for patient‑level split.

    Returns:
        (train_loader, val_loader) or (train_loader, val_loader, test_loader).
    """
    # ── Create dataset ────────────────────────────────────────────────
    if prepatched_dir is not None:
        print("🔹 Using PRE-PATCHED dataset")
        dataset = MedicalImageDataset(
            data_dir   = prepatched_dir,
            modality   = modality,
            patch_size = patch_size,
            verbose    = True,
            prepatched = True,
        )
        if metadata_csv and os.path.exists(metadata_csv):
            import pandas as pd
            df = pd.read_csv(metadata_csv)
            patient_to_files = df.groupby('original_scan')['filename'].apply(list).to_dict()
            file_to_idx = {os.path.basename(p): i
                           for i, p in enumerate(dataset.index)}
            patient_to_indices = {}
            for patient, files in patient_to_files.items():
                idxs = [file_to_idx[f] for f in files if f in file_to_idx]
                if idxs:
                    patient_to_indices[patient] = idxs
            original_entities = list(patient_to_indices.keys())
        else:
            patient_to_indices = dataset.original_to_patches
            original_entities  = list(patient_to_indices.keys())
            metadata_csv       = None
    else:
        print("🔹 Using ON-THE-FLY patching dataset")
        dataset = MedicalImageDataset(
            data_dir   = data_dir,
            modality   = modality,
            patch_size = patch_size,
            verbose    = True,
            prepatched = False,
        )
        if metadata_csv and os.path.exists(metadata_csv):
            import pandas as pd
            df = pd.read_csv(metadata_csv)
            patient_to_files = df.groupby('original_scan')['filename'].apply(list).to_dict()
            file_to_indices  = {}
            for idx, (path, _, _) in enumerate(dataset.index):
                file_to_indices.setdefault(os.path.basename(path), []).append(idx)
            patient_to_indices = {}
            for patient, files in patient_to_files.items():
                idxs = []
                for f in files:
                    idxs.extend(file_to_indices.get(f, []))
                if idxs:
                    patient_to_indices[patient] = idxs
            original_entities = list(patient_to_indices.keys())
        else:
            metadata_csv    = None
            file_to_indices = {}
            for idx, (path, _, _) in enumerate(dataset.index):
                file_to_indices.setdefault(path, []).append(idx)
            patient_to_indices = file_to_indices
            original_entities  = list(file_to_indices.keys())

    # ── Split ──────────────────────────────────────────────────────────
    use_patient_split = (metadata_csv is not None)
    if split_save_path and os.path.exists(split_save_path):
        print(f"♻️ Loading existing split from {split_save_path}")
        with open(split_save_path, 'r') as f:
            split = json.load(f)
        train_entities = split['train']
        val_entities   = split['val']
        test_entities  = split.get('test', [])
    else:
        print(f"🆕 Creating {'patient' if use_patient_split else 'file'}-level split (seed={seed})")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        shuffled = sorted(original_entities)
        random.shuffle(shuffled)
        n_total = len(shuffled)
        n_val   = max(1, int(val_frac * n_total))
        if test_frac > 0 and use_patient_split:
            n_test  = max(1, int(test_frac * n_total))
            n_train = n_total - n_val - n_test
            train_entities = shuffled[:n_train]
            val_entities   = shuffled[n_train:n_train + n_val]
            test_entities  = shuffled[n_train + n_val:]
        else:
            train_entities = shuffled[:n_total - n_val]
            val_entities   = shuffled[n_total - n_val:]
            test_entities  = []

        if split_save_path:
            os.makedirs(os.path.dirname(split_save_path) or '.', exist_ok=True)
            to_save = {'train': train_entities, 'val': val_entities}
            if test_entities:
                to_save['test'] = test_entities
            with open(split_save_path, 'w') as f:
                json.dump(to_save, f)

    train_idx = [i for e in train_entities for i in patient_to_indices[e]]
    val_idx   = [i for e in val_entities   for i in patient_to_indices[e]]
    test_idx  = [i for e in test_entities  for i in patient_to_indices[e]]

    # ── DataLoaders ────────────────────────────────────────────────────
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    kind = 'patient' if use_patient_split else 'file'
    if test_idx:
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)
        print(f"\n  Split ({kind}-level, seed={seed}):")
        print(f"    Train: {len(train_entities)} entities → {len(train_idx)} patches")
        print(f"    Val:   {len(val_entities)}   entities → {len(val_idx)} patches")
        print(f"    Test:  {len(test_entities)}  entities → {len(test_idx)} patches")
        return train_loader, val_loader, test_loader

    print(f"\n  Split ({kind}-level, seed={seed}):")
    print(f"    Train: {len(train_entities)} entities → {len(train_idx)} patches")
    print(f"    Val:   {len(val_entities)}   entities → {len(val_idx)} patches")
    return train_loader, val_loader
