"""Spectral vegetation detection: VARI index with morphological post-processing."""

import numpy as np
from skimage.morphology import closing, disk, remove_small_objects


def vari_mask(
    img: np.ndarray,
    threshold: float = 0.05,
    min_size: int = 500,
    closing_radius: int = 4,
) -> np.ndarray:
    """Compute a vegetation mask from an RGB uint8 image using the VARI index.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8, channels in RGB order.
    threshold : float
        VARI value above which a pixel is classified as vegetation.
    min_size : int
        Minimum connected-component size in pixels to retain (exclusive).
        At DOP20 resolution (20 cm/px), 500 px ≈ 20 m².
    closing_radius : int
        Disk radius for binary_closing to fill small canopy gaps.

    Returns
    -------
    np.ndarray
        Boolean array of shape (H, W).
    """
    r = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 2].astype(float)
    vari = (g - r) / (g + r - b + 1e-6)
    raw = vari > threshold
    mask = remove_small_objects(raw, max_size=min_size - 1)
    mask = closing(mask, disk(closing_radius))
    return mask
