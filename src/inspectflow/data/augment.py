"""Light augmentations for defect crops. Geometry stays inside the crop."""

from __future__ import annotations

import numpy as np


def augment_crop(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """image: uint8 array shaped (1, H, W) or (H, W)."""
    crop = image[0] if image.ndim == 3 else image
    if rng.random() < 0.5:
        crop = np.flip(crop, axis=1)
    if rng.random() < 0.5:
        crop = np.flip(crop, axis=0)
    delta = int(rng.integers(-20, 21))
    shifted = np.clip(crop.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(shifted)
