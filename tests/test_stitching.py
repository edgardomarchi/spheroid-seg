"""Tests for overlapping-patch stitching with logit averaging."""

from __future__ import annotations

import numpy as np
import pytest

from spheroid_seg.data.stitching import (
    compute_stride,
    extract_overlapping_tiles,
    stitch_logits,
)


def test_compute_stride_matches_formula() -> None:
    """Stride is round(patch_size * (1 - overlap)), at least 1."""
    assert compute_stride(128, 0.15) == round(128 * 0.85)
    assert compute_stride(100, 0.0) == 100
    assert compute_stride(100, 0.99) == 1


def test_every_padded_pixel_is_covered() -> None:
    """The overlapping grid covers every pixel of the padded canvas."""
    image = np.random.default_rng(42).random((200, 310)).astype(np.float32)
    tile_size = 64
    stride = compute_stride(tile_size, 0.25)
    tiles, origins, padding = extract_overlapping_tiles(image, tile_size, stride)

    padded_h = padding["padded_shape"][0]
    padded_w = padding["padded_shape"][1]
    coverage = np.zeros((padded_h, padded_w), dtype=np.int32)
    for y, x in origins:
        coverage[y : y + tile_size, x : x + tile_size] += 1

    assert np.all(coverage >= 1)
    assert len(tiles) == len(origins)


@pytest.mark.parametrize("shape", [(128, 128), (100, 150), (70, 200), (50, 50)])
def test_stitched_shape_equals_input_shape(shape: tuple[int, int]) -> None:
    """The stitched mask has exactly the input image shape."""
    image = np.random.default_rng(7).random(shape).astype(np.float32)
    tile_size = 64
    stride = compute_stride(tile_size, 0.25)

    tiles, origins, padding = extract_overlapping_tiles(image, tile_size, stride)
    num_classes = 3
    logits = np.zeros((len(tiles), tile_size, tile_size, num_classes), dtype=np.float32)
    logits[..., 1] = 1.0  # constant prediction
    mask = stitch_logits(logits, origins, padding, num_classes)

    assert mask.shape == shape


def test_stitching_is_seam_free_with_constant_logits() -> None:
    """Uniform logits produce a uniform argmax everywhere."""
    image = np.random.default_rng(9).random((180, 220)).astype(np.float32)
    tile_size = 64
    stride = compute_stride(tile_size, 0.25)

    tiles, origins, padding = extract_overlapping_tiles(image, tile_size, stride)
    num_classes = 3
    logits = np.zeros((len(tiles), tile_size, tile_size, num_classes), dtype=np.float32)
    logits[..., 2] = 2.5  # strongest constant class
    mask = stitch_logits(logits, origins, padding, num_classes)

    assert np.all(mask == 2)


def test_hand_computed_logit_averaging() -> None:
    """Two overlapping tiles with known logits average exactly in the overlap."""
    tile_size = 4
    stride = 2
    image = np.zeros((4, 6), dtype=np.float32)  # small image, two horizontal tiles

    tiles, origins, padding = extract_overlapping_tiles(image, tile_size, stride)
    assert len(tiles) == 2
    assert origins == [(0, 0), (0, 2)]

    num_classes = 2
    logits = np.zeros((2, tile_size, tile_size, num_classes), dtype=np.float32)
    # Tile 0 predicts class 0 everywhere.
    logits[0, :, :, 0] = 1.0
    # Tile 1 predicts class 1 everywhere.
    logits[1, :, :, 1] = 1.0

    mask = stitch_logits(logits, origins, padding, num_classes)
    assert mask.shape == (4, 6)

    # Columns 0-1 only tile 0 -> class 0.
    np.testing.assert_array_equal(mask[:, :2], 0)
    # Columns 4-5 only tile 1 -> class 1.
    np.testing.assert_array_equal(mask[:, 4:], 1)
    # Columns 2-3 overlap -> averaged logits are equal (0.5 each), argmax picks class 0
    # because both classes have the same value and argmax returns the first index.
    np.testing.assert_array_equal(mask[:, 2:4], 0)


def test_stitching_is_deterministic() -> None:
    """Identical inputs produce identical stitched masks."""
    image = np.random.default_rng(11).random((150, 170)).astype(np.float32)
    tile_size = 64
    stride = compute_stride(tile_size, 0.2)

    tiles, origins, padding = extract_overlapping_tiles(image, tile_size, stride)
    num_classes = 3
    rng = np.random.default_rng(0)
    logits = rng.random((len(tiles), tile_size, tile_size, num_classes)).astype(np.float32)

    mask_a = stitch_logits(logits, origins, padding, num_classes)
    mask_b = stitch_logits(logits, origins, padding, num_classes)

    np.testing.assert_array_equal(mask_a, mask_b)


def test_extract_overlapping_tiles_pad_reflect_preserves_shape_info() -> None:
    """Padding metadata records the original and padded shapes."""
    image = np.random.default_rng(13).random((110, 90)).astype(np.float32)
    tile_size = 64
    stride = compute_stride(tile_size, 0.0)

    tiles, origins, padding = extract_overlapping_tiles(image, tile_size, stride)

    assert padding["original_shape"] == image.shape
    assert padding["tile_size"] == tile_size
    assert padding["stride"] == stride
    assert tiles.shape[1:] == (tile_size, tile_size)


def test_image_smaller_than_patch_is_padded_to_patch_size() -> None:
    """An image smaller than one patch is padded up to tile_size."""
    image = np.random.default_rng(15).random((32, 32)).astype(np.float32)
    tile_size = 64
    stride = compute_stride(tile_size, 0.25)

    tiles, origins, padding = extract_overlapping_tiles(image, tile_size, stride)

    assert len(tiles) == 1
    assert origins == [(0, 0)]
    assert padding["padded_shape"] == (tile_size, tile_size)
    assert tiles[0].shape == (tile_size, tile_size)
