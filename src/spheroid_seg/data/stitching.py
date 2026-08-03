"""Overlapping-patch stitching with logit averaging for full-image inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def compute_stride(tile_size: int, overlap: float) -> int:
    """Derive the tile stride from patch size and overlap fraction.

    Args:
        tile_size: Side length of each square tile.
        overlap: Overlap fraction in [0, 1).

    Returns:
        Stride as ``round(tile_size * (1 - overlap))``, at least 1.
    """
    return max(1, round(tile_size * (1 - overlap)))


def _grid_origins(size: int, tile_size: int, stride: int) -> list[int]:
    """Return tile origins along one dimension that cover the whole size."""
    if size <= tile_size:
        return [0]
    n_tiles = int(np.ceil((size - tile_size) / stride)) + 1
    return [i * stride for i in range(n_tiles)]


def _pad_widths(size: int, tile_size: int, origins: Sequence[int]) -> tuple[int, int]:
    """Return (pad_before, pad_after) so the last tile fits."""
    required = origins[-1] + tile_size
    total_pad = max(required - size, 0)
    pad_before = total_pad // 2
    pad_after = total_pad - pad_before
    return pad_before, pad_after


def extract_overlapping_tiles(
    image: np.ndarray,
    tile_size: int,
    stride: int,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, Any]]:
    """Extract a regular grid of overlapping square tiles from an image.

    The image is reflect-padded so that the grid covers the entire image; the
    last tile in each dimension may extend beyond the original image bounds.
    Images smaller than ``tile_size`` are padded up to ``tile_size``.

    Args:
        image: HxW or HxWxC array.
        tile_size: Side length of each square tile.
        stride: Step between adjacent tile origins.

    Returns:
        Tiles array of shape ``(N, tile_size, tile_size[, C])``, a list of
        ``(y, x)`` origins in the padded coordinate system, and a padding
        metadata dictionary.
    """
    if image.ndim not in {2, 3}:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")

    h, w = image.shape[:2]
    y_origins = _grid_origins(h, tile_size, stride)
    x_origins = _grid_origins(w, tile_size, stride)
    pad_top, pad_bottom = _pad_widths(h, tile_size, y_origins)
    pad_left, pad_right = _pad_widths(w, tile_size, x_origins)

    pad_width = (
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        if image.ndim == 3
        else ((pad_top, pad_bottom), (pad_left, pad_right))
    )
    padded = np.pad(image, pad_width, mode="reflect")

    tiles: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for y in y_origins:
        for x in x_origins:
            tile = padded[y : y + tile_size, x : x + tile_size]
            tiles.append(tile)
            origins.append((y, x))

    padding = {
        "original_shape": image.shape,
        "padded_shape": padded.shape[:2],
        "tile_size": tile_size,
        "stride": stride,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }
    return np.stack(tiles), origins, padding


def stitch_logits(
    logits: np.ndarray,
    origins: Sequence[tuple[int, int]],
    padding: dict[str, Any],
    num_classes: int,
) -> np.ndarray:
    """Average overlapping tile logits and return the argmax mask.

    Logits are accumulated into a float32 canvas and divided by a per-pixel
    count canvas; the argmax is taken once after averaging. The result is
    cropped back to the original image size.

    Args:
        logits: Array of shape ``(N, tile_size, tile_size, num_classes)``.
        origins: ``(y, x)`` origin of each tile in the padded canvas.
        padding: Padding metadata from :func:`extract_overlapping_tiles`.
        num_classes: Number of output classes.

    Returns:
        Argmax mask of shape ``(H, W)`` matching the original image.
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected 4D logits (N,H,W,C), got shape {logits.shape}")

    tile_size = padding["tile_size"]
    padded_h, padded_w = padding["padded_shape"]
    original_shape = padding["original_shape"]

    accumulator = np.zeros((padded_h, padded_w, num_classes), dtype=np.float32)
    counts = np.zeros((padded_h, padded_w), dtype=np.float32)

    for (y, x), tile_logits in zip(origins, logits, strict=False):
        accumulator[y : y + tile_size, x : x + tile_size] += tile_logits.astype(np.float32)
        counts[y : y + tile_size, x : x + tile_size] += 1.0

    averaged = accumulator / np.maximum(counts[..., np.newaxis], 1e-6)
    argmax = np.argmax(averaged, axis=-1).astype(np.uint8)

    pad_top = padding["pad_top"]
    pad_bottom = padding["pad_bottom"]
    pad_left = padding["pad_left"]
    pad_right = padding["pad_right"]

    end_h = argmax.shape[0] - pad_bottom
    end_w = argmax.shape[1] - pad_right
    cropped = argmax[pad_top:end_h, pad_left:end_w]

    if cropped.shape != original_shape[:2]:
        raise ValueError(
            f"Stitched shape {cropped.shape} does not match original {original_shape[:2]}"
        )
    return cropped
