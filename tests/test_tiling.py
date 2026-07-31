"""Tests for full-image tile/reassemble utilities."""

from __future__ import annotations

import numpy as np
import pytest

from spheroid_seg.data.tiling import extract_tiles, reassemble_from_tiles


def _make_image(shape: tuple[int, int], channels: int = 1) -> np.ndarray:
    """Create a deterministic test image."""
    rng = np.random.default_rng(42)
    if channels == 1:
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.integers(0, 256, size=(*shape, channels), dtype=np.uint8)


@pytest.mark.parametrize("shape", [(256, 256), (200, 310), (128, 128)])
def test_tile_reassemble_roundtrip_grayscale(shape: tuple[int, int]) -> None:
    """Tiling and reassembling a grayscale image returns the original array."""
    image = _make_image(shape, channels=1)
    tiles, padding = extract_tiles(image, tile_size=64)
    reassembled = reassemble_from_tiles(tiles, padding)

    np.testing.assert_array_equal(reassembled, image)


@pytest.mark.parametrize("shape", [(256, 256, 3), (180, 250, 3)])
def test_tile_reassemble_roundtrip_rgb(shape: tuple[int, int, int]) -> None:
    """Tiling and reassembling an RGB image returns the original array."""
    image = _make_image(shape[:2], channels=3)
    tiles, padding = extract_tiles(image, tile_size=64)
    reassembled = reassemble_from_tiles(tiles, padding)

    np.testing.assert_array_equal(reassembled, image)


def test_padding_is_cropped_exactly() -> None:
    """Padding added for non-divisible sizes is removed during reassembly."""
    image = _make_image((100, 150), channels=1)
    tiles, padding = extract_tiles(image, tile_size=64)
    reassembled = reassemble_from_tiles(tiles, padding)

    assert reassembled.shape == image.shape
    np.testing.assert_array_equal(reassembled, image)


def test_extract_tiles_deterministic() -> None:
    """Extracting tiles from the same image yields identical tiles."""
    image = _make_image((256, 256), channels=1)
    tiles_a, padding_a = extract_tiles(image, tile_size=64)
    tiles_b, padding_b = extract_tiles(image, tile_size=64)

    np.testing.assert_array_equal(tiles_a, tiles_b)
    assert padding_a == padding_b


def test_tiles_have_requested_size() -> None:
    """Every tile has the requested spatial size and preserves channels."""
    image = _make_image((256, 256), channels=1)
    tiles, _ = extract_tiles(image, tile_size=64)

    assert tiles.shape[1:] == (64, 64)
    assert tiles.shape[0] == 16
