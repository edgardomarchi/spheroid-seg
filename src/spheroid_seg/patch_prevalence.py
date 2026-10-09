"""Exact class-prevalence statistics for the patch arrays used in training.

``train.py`` writes ``logs/patch_class_prevalence.csv`` when a fresh run
builds its train/validation patches. The counts describe the exact
post-augmentation patch arrays the training loop consumes; they are computed
in a read-only pass over those same in-memory arrays that consumes no
randomness, so logging never affects RNG consumption, patch order,
augmentation results, batching, losses, metrics, checkpointing, or resume
behavior.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spheroid_seg.eval import CLASS_NAMES

#: File name of the prevalence CSV inside the run's ``logs/`` directory.
PREVALENCE_CSV_NAME = "patch_class_prevalence.csv"

#: Row name of the combined foreground class (loose cell union aggregate).
OBJECT_CLASS_NAME = "object"

#: Long-format CSV schema written by ``write_prevalence_csv``.
CSV_COLUMNS = [
    "split",
    "class_name",
    "pixel_count",
    "pixel_fraction",
    "patches_with_class_count",
    "patches_with_class_fraction",
    "n_patches",
    "total_pixels",
]


@dataclass(frozen=True)
class SplitPrevalence:
    """Exact per-class statistics for one split's patch array.

    Attributes:
        pixel_counts: uint64 array of shape ``(num_classes,)`` with the exact
            per-class pixel counts.
        patches_with_class: uint64 array of shape ``(num_classes,)`` counting
            patches that contain at least one pixel of each class.
        object_pixel_count: Pixels of any non-background class — the exact
            union of the loose-cell and aggregate pixel sets.
        object_patch_count: Patches containing at least one non-background
            pixel.
        n_patches: Number of patches in the split.
        total_pixels: Total pixels across all patches of the split.
    """

    pixel_counts: np.ndarray
    patches_with_class: np.ndarray
    object_pixel_count: int
    object_patch_count: int
    n_patches: int
    total_pixels: int


def patch_class_prevalence(masks: np.ndarray, num_classes: int) -> SplitPrevalence:
    """Compute exact per-class pixel and patch-presence counts for one split.

    Counts with ``numpy.bincount``, one patch at a time, iterating views over
    the patch array without copying it; peak memory stays bounded by the
    largest single patch. Empty classes keep zero counts.

    Args:
        masks: ``(N, P, P)`` integer array of model class IDs.
        num_classes: Number of model classes.

    Returns:
        SplitPrevalence with uint64 counts.

    Raises:
        ValueError: If ``masks`` is not a 3D integer array, if ``num_classes``
            is smaller than 1, or if the array carries class IDs outside
            ``[0, num_classes)``.
    """
    masks = np.asarray(masks)
    if not np.issubdtype(masks.dtype, np.integer):
        raise ValueError(f"Masks must have an integer dtype, got {masks.dtype}")
    if masks.ndim != 3:
        raise ValueError(f"Masks must be a (N, P, P) array, got shape {masks.shape}")
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if masks.size and (int(masks.min()) < 0 or int(masks.max()) >= num_classes):
        raise ValueError(
            f"Masks contain class IDs outside [0, {num_classes}); "
            "expected already-mapped model class IDs"
        )

    n_patches = int(masks.shape[0])
    per_patch_counts = np.zeros((n_patches, num_classes), dtype=np.uint64)
    for idx, patch in enumerate(masks):
        per_patch_counts[idx] = np.bincount(patch.reshape(-1), minlength=num_classes)

    pixel_counts = per_patch_counts.sum(axis=0, dtype=np.uint64)
    patches_with_class = np.count_nonzero(per_patch_counts, axis=0).astype(np.uint64)
    foreground_counts = per_patch_counts[:, 1:]
    object_pixel_count = int(foreground_counts.sum())
    object_patch_count = int(np.count_nonzero(np.any(foreground_counts > 0, axis=1)))

    return SplitPrevalence(
        pixel_counts=pixel_counts,
        patches_with_class=patches_with_class,
        object_pixel_count=object_pixel_count,
        object_patch_count=object_patch_count,
        n_patches=n_patches,
        total_pixels=int(masks.size),
    )


def _class_row_name(class_index: int) -> str:
    """CSV row name for a model class index."""
    if class_index < len(CLASS_NAMES):
        return CLASS_NAMES[class_index]
    return f"class {class_index}"


def prevalence_csv_rows(
    splits: dict[str, SplitPrevalence], num_classes: int
) -> list[dict[str, Any]]:
    """Long-format CSV rows: one row per split and model class, plus ``object``.

    ``pixel_fraction`` is relative to all pixels of that split's patch set;
    ``patches_with_class_fraction`` is relative to that split's patch count.
    The ``object`` row reports the exact union of the non-background classes:
    its pixel count is the sum of the loose-cell and aggregate pixel counts,
    and its patch count covers patches with at least one pixel of either
    class. Fractions are 0.0 (never omitted rows) when the denominator is 0.
    """
    rows: list[dict[str, Any]] = []
    for split, stats in splits.items():
        total = stats.total_pixels
        n_patches = stats.n_patches
        for class_index in range(num_classes):
            pixel_count = int(stats.pixel_counts[class_index])
            patch_count = int(stats.patches_with_class[class_index])
            rows.append(
                {
                    "split": split,
                    "class_name": _class_row_name(class_index),
                    "pixel_count": pixel_count,
                    "pixel_fraction": pixel_count / total if total else 0.0,
                    "patches_with_class_count": patch_count,
                    "patches_with_class_fraction": patch_count / n_patches if n_patches else 0.0,
                    "n_patches": n_patches,
                    "total_pixels": total,
                }
            )
        rows.append(
            {
                "split": split,
                "class_name": OBJECT_CLASS_NAME,
                "pixel_count": stats.object_pixel_count,
                "pixel_fraction": stats.object_pixel_count / total if total else 0.0,
                "patches_with_class_count": stats.object_patch_count,
                "patches_with_class_fraction": stats.object_patch_count / n_patches
                if n_patches
                else 0.0,
                "n_patches": n_patches,
                "total_pixels": total,
            }
        )
    return rows


def write_prevalence_csv(
    path: Path,
    splits: dict[str, np.ndarray],
    num_classes: int,
) -> dict[str, SplitPrevalence]:
    """Compute per-split prevalence and write the long-format CSV.

    Args:
        path: Target CSV file; its parent directory is created if needed.
        splits: Mapping of split name (``train``/``val``) to the exact
            ``(N, P, P)`` mask patch array used by the training loop.
        num_classes: Number of model classes.

    Returns:
        Mapping of split name to its SplitPrevalence, for the stdout summary.

    Raises:
        OSError: If the CSV cannot be written. Callers let this propagate so
            training fails loudly instead of running without the diagnostic.
        ValueError: If a mask array fails validation.
    """
    stats = {name: patch_class_prevalence(masks, num_classes) for name, masks in splits.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(prevalence_csv_rows(stats, num_classes))
    return stats


def prevalence_summary(splits: dict[str, SplitPrevalence], num_classes: int) -> str:
    """Concise stdout summary of per-split foreground prevalence.

    One line per split with the foreground (object) pixel fraction, each
    foreground class pixel fraction, and the object patch count over the
    split's patch count.
    """
    lines = ["Patch class prevalence:"]
    for split, stats in splits.items():
        total = stats.total_pixels
        foreground_fraction = stats.object_pixel_count / total if total else 0.0
        parts = [f"foreground={foreground_fraction:.4f}"]
        for class_index in range(1, num_classes):
            pixel_count = int(stats.pixel_counts[class_index])
            fraction = pixel_count / total if total else 0.0
            parts.append(f"{_class_row_name(class_index).split()[0]}={fraction:.4f}")
        parts.append(f"object patches={stats.object_patch_count}/{stats.n_patches}")
        lines.append(f"  {split + ':':<6} " + " ".join(parts))
    return "\n".join(lines)
