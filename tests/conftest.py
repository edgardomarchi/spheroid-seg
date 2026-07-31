"""Shared fixtures and helpers for data-pipeline tests."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from spheroid_seg.data.synthetic import write_synthetic_pair


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
        return write_synthetic_pair(
            raw_dir,
            masks_dir,
            name,
            shape=shape,
            mask_values=mask_values,
            raw_channels=raw_channels,
            dtype=dtype,
        )

    return factory
