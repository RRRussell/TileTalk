"""Macenko H&E stain normalization.

Fits per-section hematoxylin/eosin stain vectors from a sample of pixels and
maps a target section's appearance onto a reference section's stain profile, so
that frozen encoders see images with matched color statistics. Used to test
whether cross-sample transfer failure is driven by stain/batch shift.

Reference: Macenko et al., "A method for normalizing histology slides for
quantitative analysis," ISBI 2009.
"""
from __future__ import annotations

import numpy as np


def _od(rgb_uint8: np.ndarray) -> np.ndarray:
    """RGB uint8 -> optical density, shape (..., 3)."""
    return -np.log((rgb_uint8.astype(np.float32) + 1.0) / 256.0)


def fit(pixels_uint8: np.ndarray, beta: float = 0.15, alpha: float = 1.0):
    """Estimate stain matrix HE (3x2) and per-stain max concentration (2,).

    pixels_uint8: (N, 3) RGB pixel sample (foreground-heavy is fine; background
    is filtered by the OD threshold beta).
    """
    OD = _od(pixels_uint8).reshape(-1, 3)
    OD = OD[~np.any(OD < beta, axis=1)]            # drop near-white background
    if len(OD) < 100:
        OD = _od(pixels_uint8).reshape(-1, 3)
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
    V = V[:, [2, 1]]                                # top-2 eigenvectors
    if V[0, 0] < 0:
        V[:, 0] *= -1
    if V[0, 1] < 0:
        V[:, 1] *= -1
    proj = OD @ V
    ang = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(ang, alpha), np.percentile(ang, 100 - alpha)
    v1 = V @ np.array([np.cos(lo), np.sin(lo)])
    v2 = V @ np.array([np.cos(hi), np.sin(hi)])
    HE = np.array([v1, v2]).T                       # (3, 2)
    HE /= np.linalg.norm(HE, axis=0, keepdims=True) + 1e-8
    if HE[0, 0] < HE[0, 1]:                          # ensure hematoxylin first
        HE = HE[:, ::-1]
    C = np.linalg.lstsq(HE, OD.T, rcond=None)[0]     # (2, N)
    maxC = np.percentile(C, 99, axis=1)
    return HE.astype(np.float32), maxC.astype(np.float32)


def normalize_patches(patches: np.ndarray, src_HE, src_maxC, ref_HE, ref_maxC):
    """Map `patches` (N,H,W,3 uint8) from src stain profile onto ref's."""
    N, H, W, _ = patches.shape
    out = np.empty_like(patches)
    scale = (ref_maxC / (src_maxC + 1e-8)).astype(np.float32)
    for i in range(N):
        OD = _od(patches[i]).reshape(-1, 3)          # (HW, 3)
        C = np.linalg.lstsq(src_HE, OD.T, rcond=None)[0]   # (2, HW)
        C *= scale[:, None]
        ODn = ref_HE @ C                              # (3, HW)
        I = np.clip(256.0 * np.exp(-ODn.T), 0, 255).reshape(H, W, 3)
        out[i] = I.astype(np.uint8)
    return out


def sample_pixels(patches: np.ndarray, n_patches: int = 2000, seed: int = 0):
    """Pool pixels from a random subset of patches for fitting."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(patches), min(n_patches, len(patches)), replace=False)
    return patches[idx].reshape(-1, 3)
