"""H&E patch extraction around cell centroids.

The registered H&E shares the Xenium micron coordinate origin, so a cell at
micron (x, y) maps to full-resolution pixel (x / s, y / s) with s =
PhysicalSizeX (micron/pixel). We crop a square window around each centroid and
resize to the encoder's input size. Edge crops are zero-padded.

For the MVP we load the full level-0 image into RAM once (≈2.2 GB for this
slide) and crop in numpy -- far simpler and faster than tiled random reads, and
trivial given available memory.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def micron_to_pixel(coords_um: np.ndarray, pixel_size_um: float) -> np.ndarray:
    """(n,2) micron centroids -> (n,2) full-resolution pixel coords (x, y).

    For pre-registered H&E that shares the Xenium micron origin (e.g. the breast
    replicates), pixel = micron / pixel_size.
    """
    return np.asarray(coords_um, dtype=np.float64) / float(pixel_size_um)


def micron_to_pixel_affine(coords_um: np.ndarray, affine: np.ndarray,
                           invert: bool = False) -> np.ndarray:
    """Map micron centroids to H&E pixels via a 3x3 affine (10x he_imagealignment).

    affine maps homogeneous micron (x, y, 1) -> homogeneous pixel; pass
    invert=True if the CSV is given as the pixel->micron transform.
    """
    A = np.asarray(affine, dtype=np.float64)
    if invert:
        A = np.linalg.inv(A)
    n = len(coords_um)
    hom = np.hstack([np.asarray(coords_um, dtype=np.float64), np.ones((n, 1))])
    out = hom @ A.T
    return out[:, :2] / out[:, 2:3]


def crop_patch(image: np.ndarray, cx: float, cy: float,
               window: int, out_size: Optional[int] = None) -> np.ndarray:
    """Crop a `window`x`window` patch centered at pixel (cx, cy); zero-pad edges.

    image: (H, W, 3) uint8. Returns (out, out, 3) uint8 (out=out_size or window).
    """
    H, W = image.shape[:2]
    half = window // 2
    cx, cy = int(round(cx)), int(round(cy))
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + window, y0 + window

    patch = np.zeros((window, window, image.shape[2]), dtype=image.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, W), min(y1, H)
    if sx1 > sx0 and sy1 > sy0:
        patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]

    if out_size and out_size != window:
        from PIL import Image
        patch = np.asarray(Image.fromarray(patch).resize((out_size, out_size),
                                                          Image.BILINEAR))
    return patch


def extract_patches(image: np.ndarray, centers_px: np.ndarray,
                    window: int, out_size: Optional[int] = None) -> np.ndarray:
    """Crop patches for all centers -> (n, out, out, 3) uint8 array."""
    out = out_size or window
    arr = np.zeros((len(centers_px), out, out, image.shape[2]), dtype=np.uint8)
    for i, (cx, cy) in enumerate(centers_px):
        arr[i] = crop_patch(image, cx, cy, window, out_size)
    return arr
