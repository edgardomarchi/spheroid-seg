"""Non-overlapping image tiling and reassembly for full-image inference."""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_tiles(
    image: np.ndarray,
    tile_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Split an image into non-overlapping square tiles.

    Images whose dimensions are not divisible by ``tile_size`` are reflect-padded
    before tiling; the padding metadata is returned so reassembly can crop it back.

    Args:
        image: HxW or HxWxC array.
        tile_size: Side length of each square tile.

    Returns:
        Tiles array of shape ``(N, tile_size, tile_size[, C])`` and a padding
        metadata dictionary.
    """
    if image.ndim not in {2, 3}:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")

    h, w = image.shape[:2]
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    pad_width = (
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        if image.ndim == 3
        else ((pad_top, pad_bottom), (pad_left, pad_right))
    )
    padded = np.pad(image, pad_width, mode="reflect")

    ph, pw = padded.shape[:2]
    n_h = ph // tile_size
    n_w = pw // tile_size

    if image.ndim == 3:
        c = image.shape[2]
        tiles = padded.reshape(n_h, tile_size, n_w, tile_size, c)
        tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(n_h * n_w, tile_size, tile_size, c)
    else:
        tiles = padded.reshape(n_h, tile_size, n_w, tile_size)
        tiles = tiles.transpose(0, 2, 1, 3).reshape(n_h * n_w, tile_size, tile_size)

    padding = {
        "original_shape": image.shape,
        "tile_size": tile_size,
        "n_h": n_h,
        "n_w": n_w,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }
    return tiles, padding


def reassemble_from_tiles(
    tiles: np.ndarray,
    padding: dict[str, Any],
) -> np.ndarray:
    """Reassemble an image from tiles and crop any padding introduced by tiling.

    Args:
        tiles: Tiles array from :func:`extract_tiles`.
        padding: Padding metadata from :func:`extract_tiles`.

    Returns:
        Reassembled array with the original shape from the tiling call.
    """
    n_h = padding["n_h"]
    n_w = padding["n_w"]
    tile_size = padding["tile_size"]
    original_shape = padding["original_shape"]

    if tiles.ndim == 4:
        c = tiles.shape[3]
        tiles = tiles.reshape(n_h, n_w, tile_size, tile_size, c)
        tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(n_h * tile_size, n_w * tile_size, c)
    else:
        tiles = tiles.reshape(n_h, n_w, tile_size, tile_size)
        tiles = tiles.transpose(0, 2, 1, 3).reshape(n_h * tile_size, n_w * tile_size)

    h, w = original_shape[:2]
    pad_top = padding["pad_top"]
    pad_bottom = padding["pad_bottom"]
    pad_left = padding["pad_left"]
    pad_right = padding["pad_right"]

    end_h = tiles.shape[0] - pad_bottom
    end_w = tiles.shape[1] - pad_right
    cropped = tiles[pad_top:end_h, pad_left:end_w]

    if cropped.shape != original_shape:
        raise ValueError(
            f"Reassembled shape {cropped.shape} does not match original {original_shape}"
        )
    return cropped
