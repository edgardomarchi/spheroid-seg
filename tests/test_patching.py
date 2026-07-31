"""Tests for patch extraction with object oversampling."""

from __future__ import annotations

import numpy as np

from spheroid_seg.data.patching import extract_patches


def make_image_and_mask(shape: tuple[int, int] = (256, 256)) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic grayscale image and a binary-ish mask."""
    rng = np.random.default_rng(42)
    image = rng.random(shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    # Place a dense object region in the center.
    mask[64:192, 64:192] = 1
    return image, mask


def test_extract_patches_shapes() -> None:
    """Extracted patches have the requested size and count."""
    image, mask = make_image_and_mask()
    patch_size = 64
    patches_img, patches_mask = extract_patches(
        image,
        mask,
        patch_size=patch_size,
        min_object_fraction=0.05,
        object_patch_ratio=0.5,
        patches_per_image=16,
        seed=0,
    )

    assert patches_img.shape == (16, patch_size, patch_size)
    assert patches_mask.shape == (16, patch_size, patch_size)
    assert patches_mask.dtype == np.uint8


def test_object_patch_ratio_respected() -> None:
    """The fraction of object patches is close to the configured target."""
    image, mask = make_image_and_mask()
    patches_img, patches_mask = extract_patches(
        image,
        mask,
        patch_size=64,
        min_object_fraction=0.05,
        object_patch_ratio=0.8,
        patches_per_image=100,
        seed=1,
    )

    object_patches = ((patches_mask > 0).sum(axis=(1, 2)) / (64 * 64) > 0.05).sum()
    ratio = object_patches / len(patches_mask)
    assert 0.75 <= ratio <= 0.85


def test_extract_patches_deterministic() -> None:
    """Patch extraction is deterministic for a fixed seed."""
    image, mask = make_image_and_mask()
    args = {
        "patch_size": 64,
        "min_object_fraction": 0.05,
        "object_patch_ratio": 0.5,
        "patches_per_image": 16,
        "seed": 123,
    }

    patches_a_img, patches_a_mask = extract_patches(image, mask, **args)
    patches_b_img, patches_b_mask = extract_patches(image, mask, **args)

    np.testing.assert_array_equal(patches_a_img, patches_b_img)
    np.testing.assert_array_equal(patches_a_mask, patches_b_mask)
