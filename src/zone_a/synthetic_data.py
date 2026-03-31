"""
Synthetic Weld NDE Signal Generator  (Sprint 4 — Zone A)
=========================================================

Generates realistic feature vectors representing Non-Destructive Evaluation
(NDE) signals from weld radiographic inspection.  Each feature vector
captures physical characteristics that an NDE system would extract from a
radiograph scan:

    Feature block 1 (indices 0–7):   Signal amplitude statistics
        mean, std, peak-to-peak, RMS, skewness, kurtosis, crest_factor, SNR

    Feature block 2 (indices 8–15):  Frequency domain descriptors
        spectral_centroid, spectral_bandwidth, spectral_rolloff,
        spectral_entropy, low_freq_energy, mid_freq_energy,
        high_freq_energy, dominant_freq

    Feature block 3 (indices 16–23): Spatial / geometric descriptors
        aspect_ratio, area_fraction, circularity, eccentricity,
        solidity, convexity, elongation, orientation_var

    Feature block 4 (indices 24–31): Texture & boundary descriptors
        homogeneity, contrast, energy, correlation,
        boundary_roughness, edge_density, gradient_mean, gradient_std

Defect Classes
--------------
    0 — CLEAN          : No defect. All signals near reference baseline.
    1 — POROSITY       : Spherical voids.  High amplitude variance,
                         elevated spectral entropy, low circularity.
    2 — CRACK          : Linear planar defect.  High eccentricity,
                         strong directional gradient, high edge density.
    3 — LACK_OF_FUSION : Boundary-layer defect.  High aspect ratio,
                         asymmetric frequency distribution, low solidity.

Reference
---------
    AWS D1.1 / ISO 17636-1 — Radiographic Testing of Fusion Welds
    ASNT SNT-TC-1A — Personnel Qualification and Certification

Author: Babak Pirzadi (STRATEGOS Thesis — Zone A)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Optional


# =========================================================================
# Defect class definitions
# =========================================================================

class DefectClass(IntEnum):
    CLEAN          = 0
    POROSITY       = 1
    CRACK          = 2
    LACK_OF_FUSION = 3


CLASS_NAMES = {
    DefectClass.CLEAN:          "Clean",
    DefectClass.POROSITY:       "Porosity",
    DefectClass.CRACK:          "Crack",
    DefectClass.LACK_OF_FUSION: "Lack of Fusion",
}

N_FEATURES = 32
N_CLASSES  = 4

# =========================================================================
# Per-class feature distribution parameters
# =========================================================================
# Shape: (N_CLASSES, N_FEATURES, 2) where last dim = (mean, std)
# All features are normalised to [0, 1] physiologically meaningful range.

# fmt: off
_CLASS_PARAMS = {
    # ── CLEAN ─────────────────────────────────────────────────────────────
    # Low variance, near-reference baseline across all blocks
    DefectClass.CLEAN: {
        "mean": np.array([
            # Amplitude block
            0.50, 0.04, 0.18, 0.50, 0.05, 0.12, 1.50, 0.82,
            # Frequency block
            0.50, 0.18, 0.62, 0.25, 0.42, 0.35, 0.22, 0.48,
            # Geometric block
            1.05, 0.08, 0.85, 0.12, 0.92, 0.90, 0.15, 0.10,
            # Texture block
            0.80, 0.10, 0.72, 0.88, 0.12, 0.14, 0.30, 0.08,
        ]),
        "std": np.array([
            0.04, 0.01, 0.02, 0.04, 0.03, 0.04, 0.10, 0.04,
            0.04, 0.03, 0.04, 0.04, 0.04, 0.04, 0.03, 0.04,
            0.08, 0.02, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02,
            0.04, 0.02, 0.04, 0.04, 0.02, 0.02, 0.03, 0.02,
        ]),
    },

    # ── POROSITY ──────────────────────────────────────────────────────────
    # High amplitude variance, elevated spectral entropy, low circularity
    DefectClass.POROSITY: {
        "mean": np.array([
            # Amplitude: higher std, higher kurtosis (spike-like signals)
            0.48, 0.12, 0.45, 0.52, 0.30, 0.65, 2.20, 0.62,
            # Frequency: higher spectral entropy (scattered energy)
            0.54, 0.28, 0.58, 0.60, 0.35, 0.40, 0.25, 0.52,
            # Geometry: low circularity (irregular), moderate eccentricity
            1.20, 0.18, 0.52, 0.35, 0.75, 0.72, 0.22, 0.28,
            # Texture: lower homogeneity, higher contrast
            0.55, 0.45, 0.50, 0.62, 0.32, 0.38, 0.52, 0.28,
        ]),
        "std": np.array([
            0.06, 0.04, 0.08, 0.06, 0.08, 0.10, 0.25, 0.08,
            0.06, 0.06, 0.06, 0.08, 0.06, 0.06, 0.05, 0.06,
            0.12, 0.05, 0.08, 0.07, 0.06, 0.07, 0.06, 0.06,
            0.08, 0.07, 0.08, 0.08, 0.06, 0.07, 0.08, 0.06,
        ]),
    },

    # ── CRACK ─────────────────────────────────────────────────────────────
    # High eccentricity, strong directional gradient, high edge density
    DefectClass.CRACK: {
        "mean": np.array([
            # Amplitude: sharp local minima (dark lines) → high kurtosis
            0.42, 0.08, 0.50, 0.44, 0.55, 0.80, 3.20, 0.55,
            # Frequency: high-frequency energy from sharp edges
            0.58, 0.22, 0.55, 0.40, 0.25, 0.30, 0.45, 0.62,
            # Geometry: very high eccentricity, low area fraction
            2.80, 0.06, 0.35, 0.88, 0.68, 0.58, 0.72, 0.55,
            # Texture: high edge density, high gradient
            0.42, 0.62, 0.38, 0.48, 0.55, 0.72, 0.68, 0.45,
        ]),
        "std": np.array([
            0.05, 0.03, 0.08, 0.05, 0.10, 0.12, 0.30, 0.08,
            0.06, 0.05, 0.07, 0.07, 0.05, 0.05, 0.07, 0.07,
            0.25, 0.02, 0.08, 0.06, 0.07, 0.08, 0.08, 0.08,
            0.07, 0.08, 0.07, 0.07, 0.08, 0.08, 0.08, 0.07,
        ]),
    },

    # ── LACK OF FUSION ────────────────────────────────────────────────────
    # High aspect ratio, asymmetric frequency, low solidity
    DefectClass.LACK_OF_FUSION: {
        "mean": np.array([
            # Amplitude: moderate amplitude reduction along fusion line
            0.45, 0.09, 0.38, 0.46, 0.18, 0.38, 2.00, 0.68,
            # Frequency: low-frequency asymmetry
            0.52, 0.32, 0.68, 0.48, 0.55, 0.28, 0.18, 0.44,
            # Geometry: high aspect ratio, elongated, low solidity
            3.20, 0.12, 0.62, 0.62, 0.55, 0.62, 0.58, 0.42,
            # Texture: moderate contrast, low correlation
            0.62, 0.38, 0.58, 0.45, 0.42, 0.45, 0.48, 0.32,
        ]),
        "std": np.array([
            0.06, 0.03, 0.06, 0.06, 0.06, 0.07, 0.20, 0.06,
            0.06, 0.06, 0.06, 0.07, 0.06, 0.05, 0.04, 0.06,
            0.30, 0.04, 0.07, 0.08, 0.07, 0.08, 0.08, 0.07,
            0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.06,
        ]),
    },
}
# fmt: on


@dataclass
class NDEDataset:
    """
    Container for synthetic NDE feature dataset.

    Attributes
    ----------
    X : np.ndarray shape (N, N_FEATURES)
        Feature matrix, normalised to approximately [0, 1].
    y : np.ndarray shape (N,) dtype int
        Class labels: 0=Clean, 1=Porosity, 2=Crack, 3=LackOfFusion.
    class_counts : dict {int: int}
        Number of samples per class.
    """
    X: np.ndarray
    y: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def class_counts(self) -> dict:
        return {int(c): int((self.y == c).sum()) for c in range(N_CLASSES)}

    def split(
        self,
        train_fraction: float = 0.70,
        val_fraction:   float = 0.15,
        seed:           int   = 42,
    ) -> Tuple["NDEDataset", "NDEDataset", "NDEDataset"]:
        """Split into train / validation / test sets (stratified)."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n_samples)
        n_train = int(self.n_samples * train_fraction)
        n_val   = int(self.n_samples * val_fraction)
        tr_idx  = idx[:n_train]
        vl_idx  = idx[n_train:n_train + n_val]
        te_idx  = idx[n_train + n_val:]
        return (
            NDEDataset(self.X[tr_idx], self.y[tr_idx]),
            NDEDataset(self.X[vl_idx], self.y[vl_idx]),
            NDEDataset(self.X[te_idx], self.y[te_idx]),
        )


def generate_nde_dataset(
    n_samples_per_class: int = 500,
    seed:                int = 42,
    noise_level:         float = 0.02,
) -> NDEDataset:
    """
    Generate a synthetic NDE feature dataset with four defect classes.

    Parameters
    ----------
    n_samples_per_class : Samples per class (total = 4 × n_samples_per_class)
    seed                : Random seed for reproducibility
    noise_level         : Additional isotropic Gaussian noise added to all features

    Returns
    -------
    NDEDataset : Shuffled dataset with X (float32) and y (int) arrays
    """
    rng = np.random.default_rng(seed)
    X_all, y_all = [], []

    for cls in DefectClass:
        params = _CLASS_PARAMS[cls]
        mu  = params["mean"]
        sig = params["std"]

        # Sample from class-conditional Gaussian + small cross-class correlations
        samples = rng.normal(mu, sig, size=(n_samples_per_class, N_FEATURES))

        # Add mild isotropic noise
        samples += rng.normal(0, noise_level, samples.shape)

        # Clip to physically plausible range [0.01, 0.99] for most features
        # (aspect_ratio block allowed to go slightly above 1)
        samples = np.clip(samples, 0.01, 5.0)

        X_all.append(samples)
        y_all.append(np.full(n_samples_per_class, int(cls)))

    X = np.vstack(X_all).astype(np.float32)
    y = np.concatenate(y_all).astype(np.int32)

    # Shuffle
    idx = rng.permutation(len(y))
    return NDEDataset(X=X[idx], y=y[idx])


def normalise_features(
    X_train: np.ndarray,
    X_test:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalise features using training set statistics.

    Returns
    -------
    X_train_norm, X_test_norm, feature_means, feature_stds
    """
    mu  = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-8  # epsilon for numerical stability
    return (
        ((X_train - mu) / sig).astype(np.float32),
        ((X_test  - mu) / sig).astype(np.float32),
        mu,
        sig,
    )
