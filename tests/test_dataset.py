"""Tests for paired image/mask loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spheroid_seg.data.dataset import (
    SpheroidDataset,
    load_pair,
    normalize_percentile,
    remap_classes,
)


def test_normalize_percentile_scales_to_zero_one() -> None:
    """Percentile normalization maps the requested range to [0, 1]."""
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    normalized = normalize_percentile(image, low=1, high=99)
    assert normalized.dtype == np.float32
    assert 0.0 <= normalized.min() <= normalized.max() <= 1.0


def test_remap_classes_default_mapping() -> None:
    """Default class mapping merges spheroid and organoid into aggregate."""
    mask = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.uint8)
    mapped = remap_classes(mask, {0: 0, 1: 1, 2: 2, 3: 2})
    expected = np.array([[0, 1, 2], [2, 2, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(mapped, expected)


def test_load_pair_grayscale(
    tmp_raw_mask_dirs: tuple[Path, Path], make_pair: callable
) -> None:
    """Loading a grayscale pair returns a normalized 2D image and remapped mask."""
    raw_dir, masks_dir = tmp_raw_mask_dirs
    make_pair("img1", raw_channels="grayscale")

    image, mask = load_pair(
        raw_dir / "img1.png",
        masks_dir / "img1.png",
        input_channels="grayscale",
        class_mapping={0: 0, 1: 1, 2: 2, 3: 2},
    )

    assert image.dtype == np.float32
    assert image.ndim == 2
    assert 0.0 <= image.min() <= image.max() <= 1.0
    assert mask.dtype == np.uint8
    assert mask.shape == image.shape
    assert set(np.unique(mask)).issubset({0, 1, 2})


def test_load_pair_rgb(
    tmp_raw_mask_dirs: tuple[Path, Path], make_pair: callable
) -> None:
    """Loading with input_channels='rgb' returns a 3-channel image."""
    raw_dir, masks_dir = tmp_raw_mask_dirs
    make_pair("img1", raw_channels="rgb")

    image, mask = load_pair(
        raw_dir / "img1.png",
        masks_dir / "img1.png",
        input_channels="rgb",
        class_mapping={0: 0, 1: 1, 2: 2, 3: 2},
    )

    assert image.dtype == np.float32
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert mask.ndim == 2


def test_spheroid_dataset_collects_pairs(
    tmp_raw_mask_dirs: tuple[Path, Path], make_pair: callable
) -> None:
    """The dataset pairs raw images and masks by base name."""
    for name in ("a", "b", "c"):
        make_pair(name)

    dataset = SpheroidDataset(
        tmp_raw_mask_dirs[0],
        tmp_raw_mask_dirs[1],
        input_channels="grayscale",
        class_mapping={0: 0, 1: 1, 2: 2, 3: 2},
    )

    assert len(dataset) == 3
    names = {dataset[idx][2] for idx in range(len(dataset))}
    assert names == {"a", "b", "c"}


def test_spheroid_dataset_unmatched_files_ignored(
    tmp_raw_mask_dirs: tuple[Path, Path], make_pair: callable
) -> None:
    """Raw images without a matching mask are ignored, and vice versa."""
    raw_dir, masks_dir = tmp_raw_mask_dirs
    make_pair("paired")
    make_pair("only_raw")
    (masks_dir / "only_raw.png").unlink()

    dataset = SpheroidDataset(raw_dir, masks_dir, input_channels="grayscale")
    assert len(dataset) == 1
    assert dataset[0][2] == "paired"
