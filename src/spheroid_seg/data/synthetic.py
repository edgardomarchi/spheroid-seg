"""Deterministic synthetic raw/mask pair generation for smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np


def write_synthetic_pair(
    raw_dir: Path | str,
    masks_dir: Path | str,
    name: str,
    shape: tuple[int, int] = (128, 128),
    mask_values: tuple[int, ...] = (0, 1, 2, 3),
    raw_channels: Literal["grayscale", "rgb"] = "grayscale",
    dtype: np.dtype = np.uint8,
) -> tuple[np.ndarray, np.ndarray]:
    """Create and save a deterministic raw/mask pair.

    Args:
        raw_dir: Directory for the raw image.
        masks_dir: Directory for the mask image.
        name: Base name for both files.
        shape: (height, width) of the synthetic images.
        mask_values: Values to paint into the mask.
        raw_channels: Whether the raw image is grayscale or RGB.
        dtype: Dtype of the raw image.

    Returns:
        The generated (raw, mask) arrays.
    """
    raw_dir = Path(raw_dir)
    masks_dir = Path(masks_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(hash(name) % 2**32)

    if raw_channels == "grayscale":
        raw = rng.integers(0, 256, size=shape, dtype=np.uint8).astype(dtype)
    else:
        raw = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8).astype(dtype)

    mask = np.zeros(shape, dtype=np.uint8)
    rows, cols = shape
    for idx, value in enumerate(mask_values):
        y_start = idx * rows // len(mask_values)
        y_end = (idx + 1) * rows // len(mask_values)
        mask[y_start:y_end, :] = value

    raw_path = raw_dir / f"{name}.png"
    mask_path = masks_dir / f"{name}.png"

    cv2.imwrite(str(raw_path), raw if raw.ndim == 2 else cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask)

    return raw, mask


def generate_synthetic_dataset(
    raw_dir: Path | str,
    masks_dir: Path | str,
    n_images: int = 8,
    shape: tuple[int, int] = (512, 512),
    seed: int = 42,
) -> tuple[Path, Path]:
    """Generate a deterministic synthetic dataset with blob-shaped objects.

    The masks contain three classes: background (0), loose cell (1), and
    aggregate (2). Class 3 is intentionally omitted so the default
    {0:0, 1:1, 2:2, 3:2} mapping yields the standard 3-class task.

    Args:
        raw_dir: Directory for raw images.
        masks_dir: Directory for masks.
        n_images: Number of image/mask pairs to create.
        shape: (height, width) of each image.
        seed: Random seed for reproducibility.

    Returns:
        The (raw_dir, masks_dir) paths.
    """
    raw_dir = Path(raw_dir)
    masks_dir = Path(masks_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for idx in range(n_images):
        # Background is dark noise; objects are brighter so the model can learn.
        raw = rng.integers(0, 80, size=shape, dtype=np.uint8)
        mask = np.zeros(shape, dtype=np.uint8)

        # Draw large aggregates first, then smaller loose cells on top so both
        # classes keep visible area.
        for _ in range(4):
            cy, cx = rng.integers(100, shape[0] - 100, size=2)
            radius = rng.integers(70, 120)
            cv2.circle(mask, (int(cx), int(cy)), int(radius), 2, -1)

        for _ in range(6):
            cy, cx = rng.integers(80, shape[0] - 80, size=2)
            radius = rng.integers(30, 70)
            cv2.circle(mask, (int(cx), int(cy)), int(radius), 1, -1)

        raw = np.where(mask == 1, rng.integers(120, 180, size=shape, dtype=np.uint8), raw)
        raw = np.where(mask == 2, rng.integers(180, 255, size=shape, dtype=np.uint8), raw)

        cv2.imwrite(str(raw_dir / f"synth_{idx:03d}.png"), raw)
        cv2.imwrite(str(masks_dir / f"synth_{idx:03d}.png"), mask)

    return raw_dir, masks_dir
