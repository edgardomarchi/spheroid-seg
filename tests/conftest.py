"""Shared fixtures and helpers for data-pipeline tests."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pytest


def _write_synthetic_pair(
    raw_dir: Path,
    masks_dir: Path,
    name: str,
    shape: tuple[int, int] = (128, 128),
    mask_values: tuple[int, ...] = (0, 1, 2, 3),
    raw_channels: Literal["grayscale", "rgb"] = "grayscale",
    dtype: np.dtype = np.uint8,
) -> tuple[np.ndarray, np.ndarray]:
    """Create and save a deterministic raw/mask pair for testing.

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


@pytest.fixture
def tmp_raw_mask_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Return temporary raw and mask directories."""
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    return raw_dir, masks_dir


@pytest.fixture
def make_pair(tmp_path: Path):
    """Return a factory that writes synthetic raw/mask pairs into temp dirs."""
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"

    def factory(
        name: str,
        shape: tuple[int, int] = (128, 128),
        mask_values: tuple[int, ...] = (0, 1, 2, 3),
        raw_channels: Literal["grayscale", "rgb"] = "grayscale",
        dtype: np.dtype = np.uint8,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _write_synthetic_pair(
            raw_dir,
            masks_dir,
            name,
            shape=shape,
            mask_values=mask_values,
            raw_channels=raw_channels,
            dtype=dtype,
        )

    return factory
