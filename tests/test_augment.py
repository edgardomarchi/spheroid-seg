"""Tests for the Albumentations augmentation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from spheroid_seg.data.augment import apply_augmentation, build_augmentation


@pytest.fixture
def augment_config() -> dict:
    """Minimal augmentation config for deterministic tests."""
    return {
        "flip_probability": 0.5,
        "rotate90_probability": 0.5,
        "elastic_probability": 0.3,
        "scale_probability": 0.5,
        "blur_probability": 0.2,
        "noise_probability": 0.2,
        "brightness_contrast_probability": 0.5,
        "scale_range": [0.5, 2.0],
        "elastic_alpha": 1,
        "elastic_sigma": 50,
        "blur_limit": [3, 7],
        "blur_sigma_limit": [0.1, 2.0],
        "noise_std_range": [0.01, 0.05],
        "brightness_limit": 0.3,
        "contrast_limit": 0.3,
    }


def test_augmentation_preserves_shapes(augment_config: dict) -> None:
    """Augmentation keeps image and mask shapes unchanged."""
    rng = np.random.default_rng(7)
    image = rng.random((128, 128, 3)).astype(np.float32)
    mask = rng.integers(0, 3, size=(128, 128), dtype=np.uint8)

    transform = build_augmentation(augment_config, seed=42)
    aug_image, aug_mask = apply_augmentation(transform, image, mask)

    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape
    assert aug_mask.dtype == np.uint8


def test_augmentation_preserves_mask_classes(augment_config: dict) -> None:
    """Mask pixel values remain within the original class set."""
    rng = np.random.default_rng(8)
    image = rng.random((64, 64, 3)).astype(np.float32)
    mask = rng.integers(0, 3, size=(64, 64), dtype=np.uint8)

    transform = build_augmentation(augment_config, seed=43)
    _, aug_mask = apply_augmentation(transform, image, mask)

    assert set(np.unique(aug_mask)).issubset({0, 1, 2})


def test_affine_scale_out_reflects_image_and_mask_consistently():
    """Regression: scale-out must reflect-pad without black borders.

    The mask reflects with the image so labels stay consistent.
    """
    config = {"scale_probability": 1.0, "scale_range": [0.5, 0.5]}
    transform = build_augmentation(config, seed=42)
    image = np.full((128, 128), 200, dtype=np.uint8)
    mask = np.full((128, 128), 1, dtype=np.uint8)
    aug_img, aug_mask = apply_augmentation(transform, image, mask)
    assert aug_img[0, 0] > 100  # corners stay bright (no black padding)
    assert aug_mask[0, 0] == 1  # mask reflected consistently with the image
