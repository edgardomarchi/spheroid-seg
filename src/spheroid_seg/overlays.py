"""OpenCV-based overlay grid generation for evaluation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Fixed class colormap in OpenCV BGR order. Background is black, loose cells are
# green, and aggregates are yellow.
CLASS_COLORMAP: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),  # background
    1: (0, 255, 0),  # loose cell
    2: (0, 255, 255),  # aggregate
}

_FALSE_POSITIVE_COLOR = (0, 0, 255)  # red in BGR
_FALSE_NEGATIVE_COLOR = (255, 0, 0)  # blue in BGR
_LEGEND_TEXT_COLOR = (255, 255, 255)  # white


def _gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a 2D grayscale image to a 3-channel BGR image."""
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Apply the fixed class colormap to a label mask."""
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORMAP.items():
        colored[mask == class_id] = color
    return colored


def _build_error_overlay(
    raw: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Create an error overlay on top of the raw image.

    True positives for classes 1 and 2 are shown in the class color. False
    positives are shown in red and false negatives in blue. Background true
    negatives keep the raw grayscale appearance.
    """
    overlay = _gray_to_rgb(raw.copy())

    for class_id in sorted(CLASS_COLORMAP.keys()):
        if class_id == 0:
            # Background true negatives remain as the raw image.
            continue

        tp_mask = (ground_truth == class_id) & (prediction == class_id)
        fp_mask = (ground_truth != class_id) & (prediction == class_id)
        fn_mask = (ground_truth == class_id) & (prediction != class_id)

        overlay[tp_mask] = CLASS_COLORMAP[class_id]
        overlay[fp_mask] = _FALSE_POSITIVE_COLOR
        overlay[fn_mask] = _FALSE_NEGATIVE_COLOR

    return overlay


def _add_row_label(grid: np.ndarray, row: int, panel_width: int, label: str) -> None:
    """Draw the image name and magnification label onto the raw panel."""
    y_start = row * panel_width
    roi = grid[y_start : y_start + panel_width, 0:panel_width]
    cv2.putText(
        roi,
        label,
        org=(8, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=_LEGEND_TEXT_COLOR,
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def _add_error_legend(grid: np.ndarray, row: int, panel_width: int) -> None:
    """Draw a small legend on the error overlay panel."""
    y_start = row * panel_width
    x_start = 3 * panel_width
    roi = grid[y_start : y_start + panel_width, x_start : x_start + panel_width]
    cv2.putText(
        roi,
        "FP=red FN=blue",
        org=(8, panel_width - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=_LEGEND_TEXT_COLOR,
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def select_overlay_samples(
    samples: list[dict[str, Any]],
    num_samples: int,
) -> list[dict[str, Any]]:
    """Select up to ``num_samples`` images deterministically, stratified by mag.

    The selection tries to return an equal number of 4x and 10x images, sorted
    by name within each group. If one group is smaller than the target half,
    the remaining slots are filled from the other group.

    Args:
        samples: List of sample dictionaries containing at least ``name`` and
            ``magnification`` keys.
        num_samples: Maximum number of samples to return.

    Returns:
        Deterministic, stratified subset of samples.
    """
    if not samples or num_samples <= 0:
        return []

    by_mag: dict[str, list[dict[str, Any]]] = {"4x": [], "10x": [], "unknown": []}
    for sample in samples:
        mag = sample.get("magnification", "unknown")
        by_mag.setdefault(mag, []).append(sample)

    for mag in by_mag:
        by_mag[mag].sort(key=lambda s: s["name"])

    target_per_known = num_samples // 2
    selected: list[dict[str, Any]] = []
    selected.extend(by_mag["4x"][:target_per_known])
    selected.extend(by_mag["10x"][:target_per_known])

    remaining = num_samples - len(selected)
    if remaining > 0:
        extras: list[dict[str, Any]] = []
        extras.extend(by_mag["4x"][target_per_known:])
        extras.extend(by_mag["10x"][target_per_known:])
        extras.extend(by_mag["unknown"])
        extras.sort(key=lambda s: s["name"])
        selected.extend(extras[:remaining])

    selected.sort(key=lambda s: s["name"])
    return selected


def build_overlay_grid(
    samples: list[dict[str, Any]],
    panel_width: int,
) -> np.ndarray:
    """Assemble a grid of overlay panels using OpenCV.

    Columns are ``[raw | ground truth | prediction | error overlay]``. Each row
    is labeled with the image name and magnification. The error overlay includes
    a small legend.

    Args:
        samples: List of sample dictionaries with keys ``raw`` (grayscale uint8),
            ``gt`` (uint8 label mask), ``pred`` (uint8 label mask), ``name``,
            and ``magnification``.
        panel_width: Width and height of each square panel in pixels.

    Returns:
        BGR uint8 grid image of shape ``(N * panel_width, 4 * panel_width, 3)``.
    """
    n_rows = len(samples)
    grid = np.zeros((n_rows * panel_width, 4 * panel_width, 3), dtype=np.uint8)

    for row, sample in enumerate(samples):
        raw = sample["raw"]
        gt = sample["gt"]
        pred = sample["pred"]

        panels = [
            _gray_to_rgb(raw),
            _colorize_mask(gt),
            _colorize_mask(pred),
            _build_error_overlay(raw, gt, pred),
        ]

        for col, panel in enumerate(panels):
            resized = cv2.resize(panel, (panel_width, panel_width))
            x_start = col * panel_width
            y_start = row * panel_width
            grid[y_start : y_start + panel_width, x_start : x_start + panel_width] = resized

        label = f"{sample['name']} ({sample['magnification']})"
        _add_row_label(grid, row, panel_width, label)
        _add_error_legend(grid, row, panel_width)

    return grid
