"""Validation diagnostics: prevalence, false-positive components, bias sweep.

Three read-only analyses over a selected evaluation split (normally
``val``) that explain the current model's foreground overprediction before
any training change is considered:

1. Exact class-pixel prevalence in full-image train/val masks and in the
   saved augmented patches of a training run (``training_patches.npz``).
2. Connected-component decomposition of the baseline checkpoint's
   false-positive pixels (object / loose cell / aggregate analyses).
3. A post-hoc background-logit-bias sweep computed from the model
   probabilities without retraining.

The command is streaming by design: each image is tiled, predicted once,
folded into small accumulators (uint32 confusion counts, float64 proper
-score partial sums, per-component scalar statistics), and released.
Per-pixel probability maps are never retained beyond the current tile.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Same allocator guard as eval.py: keep glibc malloc arenas bounded so the
# streaming loop reuses freed blocks across images.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import jax.numpy as jnp
import numpy as np
import scipy.ndimage

from spheroid_seg import eval as eval_module
from spheroid_seg.data.dataset import _read_mask, has_real_pairs, load_pair, remap_classes
from spheroid_seg.data.metadata import parse_magnification
from spheroid_seg.data.splits import load_split_list
from spheroid_seg.data.synthetic import generate_synthetic_dataset, synthetic_split_names
from spheroid_seg.data.tiling import extract_tiles, reassemble_from_tiles
from spheroid_seg.metrics import (
    add_proper_score_sums,
    empty_proper_score_sums,
    finalize_proper_scores,
    proper_score_partial_sums,
)

CLASS_NAMES = eval_module.CLASS_NAMES

# Post-hoc background biases and component area thresholds (pixels). These are
# diagnostic parameters with stable defaults, overridable via the CLI.
DEFAULT_BACKGROUND_BIAS_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT_AREA_THRESHOLDS = (16, 64, 256, 1024, 4096)

# False-positive analyses: any foreground pixel outside ground-truth
# foreground ("object"), plus one-vs-rest analyses per foreground class.
ANALYSIS_MASKS = ("object", "loose cell", "aggregate")

# Keys of the training_patches.npz written by train.py::save_training_state.
_PATCH_KEYS = {"train": ("train_images", "train_masks"), "val": ("val_images", "val_masks")}


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _cell(value: Any) -> Any:
    """CSV cell: empty string for NaN/undefined, repr for floats, str otherwise."""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (np.floating,)):
        f = float(value)
        return "" if math.isnan(f) else repr(f)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows([[_cell(v) for v in row] for row in rows])


# ---------------------------------------------------------------------------
# Part 1 — class prevalence (full images and saved patches)
# ---------------------------------------------------------------------------


def class_pixel_counts(
    mask: np.ndarray, num_classes: int, class_mapping: dict[int, int] | None = None
) -> np.ndarray:
    """Exact per-class pixel counts of a single mask.

    Args:
        mask: 2D integer mask array.
        num_classes: Number of model classes.
        class_mapping: Optional raw-ID -> model-class mapping (the repository
            mapping merges original IDs 2 and 3 into model class 2).

    Returns:
        uint64 array of shape ``(num_classes,)``.

    Raises:
        ValueError: If the mask is not integer-typed or contains IDs outside
            ``[0, num_classes)`` after mapping.
    """
    mask = np.asarray(mask)
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Mask must have an integer dtype, got {mask.dtype}")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    if class_mapping is not None:
        if mask.size and (int(mask.min()) < 0 or int(mask.max()) > 255):
            raise ValueError("Mask IDs must be in [0, 255] to apply the class mapping")
        mask = remap_classes(mask.astype(np.uint8), class_mapping)
    counts = np.bincount(mask.ravel(), minlength=num_classes)
    if counts.shape[0] != num_classes:
        raise ValueError(
            f"Mask contains class ID {int(mask.max())}, outside "
            f"[0, {num_classes}); check the class mapping"
        )
    return counts.astype(np.uint64)


def _synthetic_mask_names(config: dict[str, Any], split: str) -> list[str]:
    """Return the deterministic synthetic split assignment for ``split``."""
    n_images = config.get("synthetic_n_images", 16)
    return synthetic_split_names(n_images, config["seed"])[split]


def full_image_prevalence(
    config: dict[str, Any],
    splits: tuple[str, ...] = ("train", "val"),
    tmp_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Exact class-pixel prevalence of the full-image masks, one image at a time.

    Uses the real masks listed in ``data/splits/<split>.txt`` when real
    raw/mask pairs exist; otherwise falls back to the deterministic synthetic
    dataset (the same fallback the eval path uses).

    Returns:
        Mapping ``split -> {"counts": uint64 (C,), "n_pixels": int}``.
    """
    num_classes = config["num_classes"]
    class_mapping = config.get("class_mapping")
    raw_dir = Path(config["data"]["raw_dir"])
    masks_dir = Path(config["data"]["masks_dir"])
    splits_dir = Path(config["data"]["splits_dir"])

    prevalence: dict[str, dict[str, Any]] = {}

    def accumulate(split: str, mask_paths: list[Path]) -> None:
        counts = np.zeros(num_classes, dtype=np.uint64)
        n_pixels = 0
        for mask_path in mask_paths:
            mask_counts = class_pixel_counts(_read_mask(mask_path), num_classes, class_mapping)
            counts += mask_counts
            n_pixels += int(mask_counts.sum())
        prevalence[split] = {"counts": counts, "n_pixels": n_pixels}

    if has_real_pairs(raw_dir, masks_dir):
        for split in splits:
            names = load_split_list(splits_dir, split)
            mask_paths = []
            for name in names:
                mask_path = eval_module._find_file_with_stem(masks_dir, name)
                if mask_path is None:
                    raise FileNotFoundError(
                        f"Mask '{name}' from {split}.txt not found in {masks_dir}"
                    )
                mask_paths.append(mask_path)
            accumulate(split, mask_paths)
        return prevalence

    # Synthetic fallback: generate the same dataset training uses.
    if tmp_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            return full_image_prevalence(config, splits, Path(tmp))
    synth_raw = tmp_dir / "raw"
    synth_masks = tmp_dir / "masks"
    generate_synthetic_dataset(
        synth_raw,
        synth_masks,
        n_images=config.get("synthetic_n_images", 16),
        shape=config.get("synthetic_image_shape", (512, 512)),
        seed=config["seed"],
    )
    for split in splits:
        mask_paths = []
        for name in _synthetic_mask_names(config, split):
            mask_path = eval_module._find_file_with_stem(synth_masks, name)
            if mask_path is None:  # pragma: no cover - defensive
                raise FileNotFoundError(f"Synthetic mask '{name}' not found in {synth_masks}")
            mask_paths.append(mask_path)
        accumulate(split, mask_paths)
    return prevalence


def saved_patch_prevalence(npz_path: Path, num_classes: int) -> dict[str, dict[str, Any]]:
    """Exact class-pixel prevalence of a run's saved augmented patches.

    The NPZ written by ``train.py::save_training_state`` holds the exact
    train/validation patch arrays the loop used. Its schema is validated
    before use: a missing file or an uninterpretable schema fails with a
    clear error instead of silently substituting a reconstructed patch set.

    Arrays are processed one at a time and released; peak memory stays
    bounded by the largest single patch array.

    Returns:
        Mapping ``split -> {"counts": uint64 (C,), "n_pixels": int,
        "n_patches": int}``.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Saved training patches not found: {npz_path}. The saved patch "
            "prevalence requires the run's checkpoints/training_patches.npz; "
            "re-run with --skip-saved-patch-prevalence to analyze full images only."
        )

    prevalence: dict[str, dict[str, Any]] = {}
    with np.load(npz_path, allow_pickle=False) as npz:
        missing = [key for pair in _PATCH_KEYS.values() for key in pair if key not in npz.files]
        if missing:
            raise ValueError(
                f"training_patches.npz at {npz_path} does not match the expected "
                f"schema (keys {sorted(_PATCH_KEYS.values())}); missing {missing}. "
                "Refusing to substitute a reconstructed patch set."
            )
        for split, (images_key, masks_key) in _PATCH_KEYS.items():
            images = npz[images_key]
            if images.ndim not in (3, 4):
                raise ValueError(
                    f"training_patches.npz at {npz_path}: '{images_key}' must be a "
                    f"(N, P, P[, C]) array, got shape {images.shape}"
                )
            n_patches = len(images)
            del images
            masks = np.asarray(npz[masks_key])
            if masks.ndim != 3:
                raise ValueError(
                    f"training_patches.npz at {npz_path}: '{masks_key}' must be a "
                    f"(N, P, P) array, got shape {masks.shape}"
                )
            if not np.issubdtype(masks.dtype, np.integer):
                raise ValueError(
                    f"training_patches.npz at {npz_path}: '{masks_key}' must have an "
                    f"integer dtype, got {masks.dtype}"
                )
            if len(masks) != n_patches:
                raise ValueError(
                    f"training_patches.npz at {npz_path}: '{images_key}' has "
                    f"{n_patches} patches but '{masks_key}' has {len(masks)}"
                )
            if masks.size == 0:
                counts = np.zeros(num_classes, dtype=np.uint64)
            else:
                if int(masks.min()) < 0 or int(masks.max()) >= num_classes:
                    raise ValueError(
                        f"training_patches.npz at {npz_path}: '{masks_key}' contains "
                        f"values outside [0, {num_classes}); the saved patches are "
                        "expected to carry already-mapped model class IDs"
                    )
                counts = np.bincount(masks.ravel(), minlength=num_classes).astype(np.uint64)
            prevalence[split] = {
                "counts": counts,
                "n_pixels": int(counts.sum()),
                "n_patches": int(n_patches),
            }
    return prevalence


def _prevalence_rows(
    full: dict[str, dict[str, Any]],
    saved: dict[str, dict[str, Any]] | None,
    splits: tuple[str, ...],
) -> list[list[Any]]:
    """Long-format rows for patch_prevalence.csv."""
    rows: list[Any] = []
    for source, table in (("full_image", full), ("saved_patch", saved)):
        if table is None:
            continue
        for split in splits:
            if split not in table:
                continue
            counts = table[split]["counts"]
            total = table[split]["n_pixels"]
            for idx, class_name in enumerate(CLASS_NAMES[: len(counts)]):
                fraction = float(counts[idx]) / total if total else 0.0
                rows.append([source, split, class_name, int(counts[idx]), fraction])
    return rows


# ---------------------------------------------------------------------------
# Part 2 — false-positive connected components
# ---------------------------------------------------------------------------


def false_positive_mask(prediction: np.ndarray, target: np.ndarray, analysis: str) -> np.ndarray:
    """Boolean false-positive mask for one analysis definition.

    Args:
        prediction: Argmax prediction mask (HxW, model class IDs).
        target: Ground-truth mask (HxW, model class IDs).
        analysis: One of ``ANALYSIS_MASKS``:
            - ``object``: predicted foreground (class 1 or 2) where the
              ground truth is background;
            - ``loose cell``: predicted class 1 where the ground truth is
              not class 1;
            - ``aggregate``: predicted class 2 where the ground truth is
              not class 2.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if analysis == "object":
        return (prediction >= 1) & (target == 0)
    if analysis == "loose cell":
        return (prediction == 1) & (target != 1)
    if analysis == "aggregate":
        return (prediction == 2) & (target != 2)
    raise ValueError(f"Unknown analysis mask '{analysis}'; expected one of {ANALYSIS_MASKS}")


def component_table(fp_mask: np.ndarray, image: np.ndarray) -> list[dict[str, Any]]:
    """Per-component statistics of a false-positive mask (8-connectivity).

    Args:
        fp_mask: 2D boolean mask of false-positive pixels.
        image: Normalized image (HxW or HxWxC) used for intensity statistics;
            multi-channel images are reduced to their channel mean.

    Returns:
        One dictionary per component with keys ``component_id`` (1-based,
        label order), ``area``, ``bbox_min_row``/``bbox_min_col``/
        ``bbox_max_row``/``bbox_max_col`` (inclusive), ``centroid_row`` /
        ``centroid_col``, ``min_border_distance`` (pixels to the nearest
        image border), ``touches_border``, ``mean_intensity`` and
        ``median_intensity`` (normalized image values inside the component).
    """
    fp_mask = np.asarray(fp_mask, dtype=bool)
    if fp_mask.ndim != 2:
        raise ValueError(f"fp_mask must be 2D, got shape {fp_mask.shape}")
    intensity = np.asarray(image, dtype=np.float64)
    if intensity.ndim == 3:
        intensity = intensity.mean(axis=-1)
    if intensity.shape != fp_mask.shape:
        raise ValueError(
            f"intensity shape {intensity.shape} does not match fp_mask shape {fp_mask.shape}"
        )

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity, as documented
    labels, n_components = scipy.ndimage.label(fp_mask, structure=structure)
    if n_components == 0:
        return []

    height, width = fp_mask.shape
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=n_components + 1)[1:]
    intensity_sums = np.bincount(flat, weights=intensity.ravel(), minlength=n_components + 1)[1:]
    row_grid = np.broadcast_to(np.arange(height, dtype=np.float64)[:, None], fp_mask.shape).ravel()
    col_grid = np.broadcast_to(np.arange(width, dtype=np.float64)[None, :], fp_mask.shape).ravel()
    row_sums = np.bincount(flat, weights=row_grid, minlength=n_components + 1)[1:]
    col_sums = np.bincount(flat, weights=col_grid, minlength=n_components + 1)[1:]
    slices = scipy.ndimage.find_objects(labels)

    rows = []
    for idx in range(n_components):
        sl = slices[idx]
        r0, r1 = sl[0].start, sl[0].stop - 1
        c0, c1 = sl[1].start, sl[1].stop - 1
        area = int(areas[idx])
        component_median = float(np.median(intensity[sl][labels[sl] == (idx + 1)]))
        rows.append(
            {
                "component_id": idx + 1,
                "area": area,
                "bbox_min_row": int(r0),
                "bbox_min_col": int(c0),
                "bbox_max_row": int(r1),
                "bbox_max_col": int(c1),
                "centroid_row": float(row_sums[idx]) / area,
                "centroid_col": float(col_sums[idx]) / area,
                "min_border_distance": int(min(r0, c0, height - 1 - r1, width - 1 - c1)),
                "touches_border": bool(min(r0, c0, height - 1 - r1, width - 1 - c1) == 0),
                "mean_intensity": float(intensity_sums[idx]) / area,
                "median_intensity": component_median,
            }
        )
    return rows


def component_summary(
    rows: list[dict[str, Any]], area_thresholds: tuple[int, ...]
) -> dict[str, Any]:
    """Aggregate component statistics for one image/analysis mask.

    The ``area_below_<t>`` keys sum the areas of components strictly below
    threshold ``t``; the matching ``frac_below_<t>`` keys divide by the total
    false-positive area (0 when there are no false positives).
    """
    areas = np.asarray([int(r["area"]) for r in rows], dtype=np.int64)
    total = int(areas.sum()) if len(areas) else 0
    summary: dict[str, Any] = {
        "total_fp_area": total,
        "n_components": len(rows),
        "mean_area": float(areas.mean()) if len(areas) else 0.0,
        "median_area": float(np.median(areas)) if len(areas) else 0.0,
        "max_area": int(areas.max()) if len(areas) else 0,
    }
    for threshold in area_thresholds:
        below = int(areas[areas < threshold].sum()) if len(areas) else 0
        summary[f"area_below_{threshold}"] = below
        summary[f"frac_below_{threshold}"] = float(below) / total if total else 0.0
    border_area = int(sum(int(r["area"]) for r in rows if r["touches_border"]))
    summary["border_fp_area"] = border_area
    summary["border_fp_fraction"] = float(border_area) / total if total else 0.0
    return summary


# ---------------------------------------------------------------------------
# Part 3 — background-logit-bias sweep
# ---------------------------------------------------------------------------


def parse_bias_grid(text: str) -> tuple[float, ...]:
    """Parse a comma-separated background-bias grid.

    Values must be finite and unique; order is preserved. An exact ``0.0``
    entry selects the bit-identical fast path (raw probabilities, no log
    transform) so bias 0 reproduces the existing eval exactly.
    """
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError("Invalid --background-bias-grid: at least one value is required")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Invalid --background-bias-grid {text!r}: not a number") from exc
    if any(math.isnan(v) or math.isinf(v) for v in values):
        raise ValueError(f"Invalid --background-bias-grid {text!r}: values must be finite")
    if len(set(values)) != len(values):
        raise ValueError(f"Invalid --background-bias-grid {text!r}: duplicate values")
    return values


def parse_area_thresholds(text: str) -> tuple[int, ...]:
    """Parse comma-separated component area thresholds (positive integers)."""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError("Invalid --area-thresholds: at least one value is required")
    values = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --area-thresholds {text!r}: '{part}' is not an integer"
            ) from exc
        if value <= 0:
            raise ValueError(f"Invalid --area-thresholds {text!r}: thresholds must be positive")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError(f"Invalid --area-thresholds {text!r}: duplicate values")
    return tuple(values)


def adjusted_prediction(probs: np.ndarray, background_bias: float) -> tuple[np.ndarray, np.ndarray]:
    """Argmax and adjusted probabilities under a background-logit bias.

    The model output is softmax probabilities, not logits. Adding ``delta``
    to the background logit is applied as the mathematically equivalent
    log-probability shift::

        adjusted_logp = log(probs) + delta * one_hot(background)
        adjusted_probs = softmax(adjusted_logp)

    so boosting the background class leaves the relative foreground order
    unchanged and can only move argmax decisions toward background. At
    ``delta == 0`` the original probabilities are returned untouched, which
    keeps the bias-0 accumulation bit-identical to the existing eval path.

    One numerical corner, mirroring exact softmax semantics: the model emits
    float32 probabilities, and a background probability can underflow to
    exactly 0.0 for extremely confident foreground tiles; ``log(0) = -inf``
    means no finite bias can flip such a pixel to background. Pixels with any
    positive background probability flip once the bias exceeds their log-odds
    gap, so a sufficiently large grid still drives the foreground count to
    the underflow-pixel floor.

    Args:
        probs: Softmax probabilities of shape ``(..., C)``.
        background_bias: Bias ``delta`` added to the background logit.

    Returns:
        Tuple ``(argmax prediction, probabilities)``; probabilities are
        float64 for non-zero biases and the original array for zero.
    """
    probs = np.asarray(probs)
    if background_bias == 0.0:
        return np.argmax(probs, axis=-1), probs
    logp = np.log(probs.astype(np.float64))  # log(0) -> -inf, handled by softmax
    logp[..., 0] += background_bias
    logp -= np.max(logp, axis=-1, keepdims=True)
    exp = np.exp(logp)
    adjusted = exp / exp.sum(axis=-1, keepdims=True)
    return np.argmax(adjusted, axis=-1), adjusted


def precision_recall_from_confusion(confusion: np.ndarray) -> dict[str, np.ndarray]:
    """Per-class precision and recall derived from a confusion matrix.

    Precision is ``TP / column sum`` and recall is ``TP / row sum``; both are
    NaN when their denominator is zero (never an invented convention).
    """
    confusion = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(confusion)
    pred_area = confusion.sum(axis=0)
    gt_area = confusion.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(pred_area > 0, tp / np.maximum(pred_area, 1), np.nan)
        recall = np.where(gt_area > 0, tp / np.maximum(gt_area, 1), np.nan)
    return {"precision": precision, "recall": recall}


def object_precision_recall_from_3x3(confusion: np.ndarray) -> tuple[float, float]:
    """Object (class 1 or 2) precision and recall from the 3x3 matrix.

    The object virtual class collapses the matrix as documented in
    ``docs/evaluation.md`` (``background = class 0``,
    ``object = class 1 | class 2``); precision/recall use the same
    column-sum/row-sum semantics as the per-class scores.
    """
    obj_conf = np.asarray(eval_module.object_confusion_from_3x3(confusion), dtype=np.float64)
    scores = precision_recall_from_confusion(obj_conf)
    return float(scores["precision"][1]), float(scores["recall"][1])


class BiasGridAccumulators:
    """Pooled confusion counts and proper-score sums for a grid of biases.

    State per bias value and per group (``overall`` plus magnification
    groups): a uint32 3x3 confusion matrix and float64 proper-score partial
    sums. Proper sums are staged per image and folded into the group
    accumulators once per image — the same nesting as the eval path — so
    bias 0 is bit-identical to :func:`spheroid_seg.eval.evaluate_split`.
    Confusion counts are exact integers, so they fold per tile.

    Usage: ``begin_image()`` before an image's tiles, ``add_tile(...)`` per
    valid tile region, ``end_image(groups)`` after the image.
    """

    def __init__(self, bias_grid: tuple[float, ...], num_classes: int) -> None:
        """Create accumulators for ``bias_grid`` with ``num_classes`` classes."""
        self.bias_grid = tuple(bias_grid)
        self.num_classes = num_classes
        self._confusions: list[dict[str, np.ndarray]] = [{} for _ in self.bias_grid]
        self._sums: list[dict[str, dict[str, Any]]] = [{} for _ in self.bias_grid]
        self._staged: list[dict[str, Any] | None] = [None for _ in self.bias_grid]

    def _confusion(self, delta_index: int, group: str) -> np.ndarray:
        matrix = self._confusions[delta_index].get(group)
        if matrix is None:
            matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.uint32)
            self._confusions[delta_index][group] = matrix
        return matrix

    def _sums_for(self, delta_index: int, group: str) -> dict[str, Any]:
        sums = self._sums[delta_index].get(group)
        if sums is None:
            sums = empty_proper_score_sums(self.num_classes)
            self._sums[delta_index][group] = sums
        return sums

    def begin_image(self) -> None:
        """Start a new image: reset the per-image proper-score staging."""
        self._staged = [None for _ in self.bias_grid]

    def add_tile(
        self,
        delta_index: int,
        groups: tuple[str, ...],
        prediction: np.ndarray,
        target: np.ndarray,
        probs_for_scores: np.ndarray,
    ) -> None:
        """Fold one valid tile region into the confusion counts and the staged sums."""
        for group in groups:
            matrix = self._confusion(delta_index, group)
            self._confusions[delta_index][group] = matrix + np.asarray(
                eval_module.accumulate_confusion_matrix(prediction, target, self.num_classes),
                dtype=np.uint32,
            )
        staged = self._staged[delta_index]
        partial = proper_score_partial_sums(probs_for_scores, target, self.num_classes)
        self._staged[delta_index] = (
            partial if staged is None else add_proper_score_sums(staged, partial)
        )

    def end_image(self, groups: tuple[str, ...]) -> None:
        """Fold the staged per-image sums into every requested group."""
        for delta_index, staged in enumerate(self._staged):
            if staged is None:
                continue
            for group in groups:
                add_proper_score_sums(self._sums_for(delta_index, group), staged)
            self._staged[delta_index] = None

    def confusion(self, delta_index: int, group: str) -> np.ndarray:
        """Return the accumulated confusion matrix for one bias and group."""
        return self._confusions[delta_index][group]

    def groups(self, delta_index: int = 0) -> list[str]:
        """Group names in output order: ``overall`` first, then sorted."""
        names = set(self._confusions[delta_index])
        ordered = [g for g in ("overall",) if g in names]
        ordered.extend(sorted(names - set(ordered)))
        return ordered

    def finalize(self, n_images: dict[str, int]) -> list[list[Any]]:
        """Reduce all accumulators to ``bias_sweep.csv`` rows (long format)."""
        if any(staged is not None for staged in self._staged):
            raise ValueError("BiasGridAccumulators: end_image() was not called")
        rows: list[list[Any]] = []
        for delta_index, delta in enumerate(self.bias_grid):
            delta_str = repr(float(delta))
            for group in self.groups(delta_index):
                confusion = self.confusion(delta_index, group)
                # Same reduction as the eval report (class_metrics_from_confusion,
                # float32 on exact uint32 counts) so bias 0 matches eval's
                # reported Dice/IoU bit for bit, not just mathematically.
                pooled = eval_module.class_metrics_from_confusion(confusion)
                dice = np.asarray(pooled["dice"])
                iou = np.asarray(pooled["iou"])
                scores = precision_recall_from_confusion(confusion)
                obj = eval_module.object_metrics_from_3x3(confusion)
                obj_precision, obj_recall = object_precision_recall_from_3x3(confusion)
                proper = finalize_proper_scores(
                    self._sums_for(delta_index, group), self.num_classes
                )
                gt_counts = confusion.sum(axis=1)
                pred_counts = confusion.sum(axis=0)
                obj_gt = int(confusion[1:, :].sum())
                obj_pred = int(confusion[:, 1:].sum())
                total = int(confusion.sum())
                n = n_images[group]
                for cls in range(self.num_classes):
                    rows.append(
                        [
                            delta_str,
                            group,
                            CLASS_NAMES[cls],
                            float(dice[cls]),
                            float(iou[cls]),
                            float(scores["precision"][cls]),
                            float(scores["recall"][cls]),
                            float(proper["brier_per_class"][cls]),
                            float(proper["log_loss_per_class"][cls]),
                            int(gt_counts[cls]),
                            int(pred_counts[cls]),
                            n,
                        ]
                    )
                rows.append(
                    [
                        delta_str,
                        group,
                        "object",
                        float(obj["dice"]),
                        float(obj["iou"]),
                        obj_precision,
                        obj_recall,
                        float("nan"),
                        float("nan"),
                        obj_gt,
                        obj_pred,
                        n,
                    ]
                )
                rows.append(
                    [
                        delta_str,
                        group,
                        "all",
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float(proper["brier"]),
                        float(proper["log_loss"]),
                        total,
                        total,
                        n,
                    ]
                )
        return rows

    def confusion_rows(self) -> list[list[Any]]:
        """Long-format confusion counts for ``bias_sweep_confusion.csv``."""
        rows: list[list[Any]] = []
        for delta_index, delta in enumerate(self.bias_grid):
            delta_str = repr(float(delta))
            for group in self.groups(delta_index):
                confusion = self.confusion(delta_index, group)
                for gt_idx, gt_name in enumerate(CLASS_NAMES[: self.num_classes]):
                    for pred_idx, pred_name in enumerate(CLASS_NAMES[: self.num_classes]):
                        rows.append(
                            [delta_str, group, gt_name, pred_name, int(confusion[gt_idx, pred_idx])]
                        )
        return rows


def sweep_full_image(
    predict_fn: Any,
    params: Any,
    batch_stats: Any,
    image: np.ndarray,
    mask: np.ndarray,
    tile_size: int,
    batch_size: int,
    num_classes: int,
    accumulators: BiasGridAccumulators,
    mag: str,
) -> np.ndarray:
    """Predict one image once and update every bias accumulator from its tiles.

    The baseline (bias 0) prediction is reassembled from full-tile argmax,
    exactly like the eval path, and returned for the component analysis.
    Proper scores and confusion counts are accumulated only on the valid
    (non-reflect-padded) region of each tile; per-tile probabilities are
    released before the next batch.

    Returns:
        The reassembled baseline argmax prediction (HxW uint8).
    """
    tiles, padding = extract_tiles(image, tile_size)
    if tiles.ndim == 3:
        tiles = tiles[..., np.newaxis]
    mask_tiles, mask_padding = extract_tiles(mask, tile_size)
    if (mask_padding["pad_top"], mask_padding["pad_left"]) != (
        padding["pad_top"],
        padding["pad_left"],
    ):
        raise ValueError("Image and mask tiling produced different padding.")

    valid_slices = eval_module._tile_valid_slices(padding)
    baseline_tiles: list[np.ndarray] = []
    accumulators.begin_image()
    for start in range(0, len(tiles), batch_size):
        batch = jnp.array(tiles[start : start + batch_size], dtype=jnp.float32)
        probs = np.asarray(predict_fn(params, batch_stats, batch))
        for offset in range(len(probs)):
            idx = start + offset
            r0, r1, c0, c1 = valid_slices[idx]
            gt_valid = mask_tiles[idx, r0:r1, c0:c1]
            probs_valid = probs[offset, r0:r1, c0:c1]
            for delta_index, delta in enumerate(accumulators.bias_grid):
                if delta == 0.0:
                    # Exact-eval path: raw probabilities, no log transform.
                    pred_valid = np.argmax(probs_valid, axis=-1)
                    probs_for_scores = probs_valid
                else:
                    pred_valid, probs_for_scores = adjusted_prediction(probs_valid, delta)
                accumulators.add_tile(
                    delta_index,
                    ("overall", mag),
                    pred_valid,
                    gt_valid,
                    probs_for_scores,
                )
        baseline_tiles.append(np.argmax(probs, axis=-1).astype(np.uint8))
    accumulators.end_image(("overall", mag))

    baseline_pred = reassemble_from_tiles(np.concatenate(baseline_tiles, axis=0), padding)
    return baseline_pred


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _make_output_dir(output_root: Path, config_stem: str) -> Path:
    """Create a unique diagnostics directory; never overwrite a non-empty one."""
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    candidate = output_root / f"{config_stem}_{timestamp}"
    suffix = 0
    while candidate.exists() and any(candidate.iterdir()):
        suffix += 1
        candidate = output_root / f"{config_stem}_{timestamp}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _foreground_fraction(counts: np.ndarray) -> float:
    total = float(counts.sum())
    return float(counts[1:].sum()) / total if total else 0.0


def _prevalence_summary_block(
    full: dict[str, dict[str, Any]],
    saved: dict[str, dict[str, Any]] | None,
    splits: tuple[str, ...],
    num_classes: int,
) -> str:
    """Markdown table of class fractions and saved/full foreground ratios."""
    lines = [
        "| split | class | full_image fraction | saved_patch fraction |",
        "|---|---|---:|---:|",
    ]
    for split in splits:
        full_counts = full[split]["counts"]
        saved_counts = saved[split]["counts"] if saved is not None else None
        for cls, class_name in enumerate(CLASS_NAMES[:num_classes]):
            full_frac = float(full_counts[cls]) / full[split]["n_pixels"]
            if saved_counts is not None and saved[split]["n_pixels"]:
                saved_frac = float(saved_counts[cls]) / saved[split]["n_pixels"]
                saved_cell = f"{saved_frac:.6f}"
            else:
                saved_cell = "n/a"
            lines.append(f"| {split} | {class_name} | {full_frac:.6f} | {saved_cell} |")
    if num_classes < 3:
        return "\n".join(lines)
    lines += [
        "",
        "| split | full_image foreground | full_image loose | full_image aggregate"
        " | saved_patch foreground | saved/full foreground ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in splits:
        full_counts = full[split]["counts"]
        full_fg = _foreground_fraction(full_counts)
        full_loose = float(full_counts[1]) / full[split]["n_pixels"]
        full_agg = float(full_counts[2]) / full[split]["n_pixels"]
        if saved is not None and saved[split]["n_pixels"]:
            saved_fg = _foreground_fraction(saved[split]["counts"])
            ratio = f"{saved_fg / full_fg:.4f}" if full_fg else "n/a"
            saved_cell = f"{saved_fg:.6f}"
        else:
            saved_cell = "n/a"
            ratio = "n/a"
        lines.append(
            f"| {split} | {full_fg:.6f} | {full_loose:.6f} | {full_agg:.6f}"
            f" | {saved_cell} | {ratio} |"
        )
    return "\n".join(lines)


def run_diagnostics(
    config: dict[str, Any],
    config_path: Path | str,
    split: str,
    checkpoint_path: Path,
    run_dir: Path | None,
    bias_grid: tuple[float, ...],
    area_thresholds: tuple[int, ...],
    output_root: Path,
    max_images: int | None = None,
    skip_saved_patch_prevalence: bool = False,
) -> Path:
    """Run all three diagnostics analyses and write their outputs.

    Images are processed one at a time; only small aggregated state
    (confusion counts, proper-score partial sums, per-component scalar
    statistics) survives across images.

    Returns:
        The unique output directory containing the CSVs and ``summary.md``.
    """
    num_classes = config["num_classes"]
    eval_config = config.get("eval", {})
    eval_batch_size = eval_config.get("batch_size", config["batch_size"])

    model, ckpt = eval_module.load_model_and_checkpoint(config, checkpoint_path)
    predict_fn = eval_module._make_predict_fn(model.apply)

    output_dir = _make_output_dir(Path(output_root), Path(config_path).stem)
    config_stem = Path(config_path).stem
    splits = ("train", "val")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ---- Part 1: prevalence -------------------------------------------
        if not has_real_pairs(Path(config["data"]["raw_dir"]), Path(config["data"]["masks_dir"])):
            print(
                "No real raw/mask pairs found; full-image prevalence uses the "
                "deterministic synthetic dataset."
            )
        full_prevalence = full_image_prevalence(config, splits=splits, tmp_dir=tmp_path)

        saved_prevalence: dict[str, dict[str, Any]] | None = None
        if skip_saved_patch_prevalence:
            print("Skipping saved-patch prevalence (--skip-saved-patch-prevalence).")
        else:
            if run_dir is None:
                raise ValueError(
                    "Saved-patch prevalence requires --run-dir (or a --checkpoint inside "
                    "a run's checkpoints/ directory) to locate training_patches.npz."
                )
            npz_path = Path(run_dir) / "checkpoints" / "training_patches.npz"
            saved_prevalence = saved_patch_prevalence(npz_path, num_classes)
        _write_csv(
            output_dir / "patch_prevalence.csv",
            ["source", "split", "class_name", "pixel_count", "pixel_fraction"],
            _prevalence_rows(full_prevalence, saved_prevalence, splits),
        )
        print(f"Part 1 (prevalence) written to {output_dir / 'patch_prevalence.csv'}")

        # ---- Parts 2 + 3: single streaming pass over the split ------------
        entries = eval_module._resolve_split_entries(config, split, tmp_path)
        if not entries:
            raise ValueError(f"No images found for split '{split}'.")
        if max_images is not None:
            entries = entries[:max_images]

        accumulators = BiasGridAccumulators(bias_grid, num_classes)
        n_images: dict[str, int] = {}
        component_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        input_channels = config["input_channels"]
        class_mapping = config["class_mapping"]

        for raw_path, mask_path, name in entries:
            image, mask = load_pair(
                raw_path,
                mask_path,
                input_channels=input_channels,
                class_mapping=class_mapping,
            )
            mag = parse_magnification(name)
            baseline_pred = sweep_full_image(
                predict_fn,
                ckpt["params"],
                ckpt["batch_stats"],
                image,
                mask,
                tile_size=config["patch_size"],
                batch_size=eval_batch_size,
                num_classes=num_classes,
                accumulators=accumulators,
                mag=mag,
            )
            n_images["overall"] = n_images.get("overall", 0) + 1
            n_images[mag] = n_images.get(mag, 0) + 1

            for analysis in ANALYSIS_MASKS:
                fp = false_positive_mask(baseline_pred, mask, analysis)
                rows = component_table(fp, image)
                for row in rows:
                    component_rows.append(
                        {
                            "image": name,
                            "magnification": mag,
                            "analysis_mask": analysis,
                            **row,
                        }
                    )
                summary_rows.append(
                    {
                        "image": name,
                        "magnification": mag,
                        "analysis_mask": analysis,
                        **component_summary(rows, area_thresholds),
                    }
                )
            # The image, mask, and baseline prediction are released here.

    # ---- Write part 2 outputs ----------------------------------------------
    _write_csv(
        output_dir / "fp_components.csv",
        [
            "image",
            "magnification",
            "analysis_mask",
            "component_id",
            "area",
            "bbox_min_row",
            "bbox_min_col",
            "bbox_max_row",
            "bbox_max_col",
            "centroid_row",
            "centroid_col",
            "min_border_distance",
            "touches_border",
            "mean_intensity",
            "median_intensity",
        ],
        [
            [
                r["image"],
                r["magnification"],
                r["analysis_mask"],
                r["component_id"],
                r["area"],
                r["bbox_min_row"],
                r["bbox_min_col"],
                r["bbox_max_row"],
                r["bbox_max_col"],
                r["centroid_row"],
                r["centroid_col"],
                r["min_border_distance"],
                r["touches_border"],
                r["mean_intensity"],
                r["median_intensity"],
            ]
            for r in component_rows
        ],
    )

    threshold_cols: list[str] = []
    for threshold in area_thresholds:
        threshold_cols.extend([f"area_below_{threshold}", f"frac_below_{threshold}"])
    summary_header = [
        "image",
        "magnification",
        "analysis_mask",
        "total_fp_area",
        "n_components",
        "mean_area",
        "median_area",
        "max_area",
        *threshold_cols,
        "border_fp_area",
        "border_fp_fraction",
    ]
    _write_csv(
        output_dir / "fp_component_summary.csv",
        summary_header,
        [[r.get(col, "") for col in summary_header] for r in summary_rows],
    )
    print(f"Part 2 (FP components) written to {output_dir / 'fp_components.csv'}")

    # ---- Write part 3 outputs ----------------------------------------------
    _write_csv(
        output_dir / "bias_sweep.csv",
        [
            "background_bias",
            "group",
            "class",
            "dice",
            "iou",
            "precision",
            "recall",
            "brier",
            "log_loss",
            "gt_pixels",
            "pred_pixels",
            "n_images",
        ],
        accumulators.finalize(n_images),
    )
    _write_csv(
        output_dir / "bias_sweep_confusion.csv",
        ["background_bias", "group", "gt", "prediction", "count"],
        accumulators.confusion_rows(),
    )
    print(f"Part 3 (bias sweep) written to {output_dir / 'bias_sweep.csv'}")

    # ---- Markdown summary ---------------------------------------------------
    _write_summary(
        output_dir / "summary.md",
        config_stem=config_stem,
        split=split,
        checkpoint_path=Path(checkpoint_path),
        run_dir=Path(run_dir) if run_dir is not None else None,
        bias_grid=bias_grid,
        area_thresholds=area_thresholds,
        max_images=max_images,
        skip_saved_patch_prevalence=skip_saved_patch_prevalence,
        full_prevalence=full_prevalence,
        saved_prevalence=saved_prevalence,
        splits=splits,
        num_classes=num_classes,
        component_rows=component_rows,
        summary_rows=summary_rows,
        accumulators=accumulators,
        n_images=n_images,
    )
    return output_dir


def _write_summary(
    path: Path,
    *,
    config_stem: str,
    split: str,
    checkpoint_path: Path,
    run_dir: Path | None,
    bias_grid: tuple[float, ...],
    area_thresholds: tuple[int, ...],
    max_images: int | None,
    skip_saved_patch_prevalence: bool,
    full_prevalence: dict[str, dict[str, Any]],
    saved_prevalence: dict[str, dict[str, Any]] | None,
    splits: tuple[str, ...],
    num_classes: int,
    component_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    accumulators: BiasGridAccumulators,
    n_images: dict[str, int],
) -> None:
    """Write the human-readable summary.md next to the CSV outputs."""
    lines = [
        "# Validation diagnostics summary",
        "",
        f"- config: `{config_stem}`",
        f"- split: `{split}`",
        f"- checkpoint: `{checkpoint_path}`",
        f"- run dir: `{run_dir}`" if run_dir is not None else "- run dir: n/a",
        f"- background bias grid: {list(bias_grid)}",
        f"- area thresholds (px): {list(area_thresholds)}",
        f"- images analyzed: {n_images.get('overall', 0)}"
        + (f" (--max-images {max_images}; smoke/debug only)" if max_images else ""),
        "",
    ]
    if skip_saved_patch_prevalence:
        lines += [
            "> Saved-patch prevalence was **skipped** via `--skip-saved-patch-prevalence`",
            "> (the run's `training_patches.npz` was unavailable).",
            "",
        ]

    lines += ["## Part 1 — class prevalence", ""]
    lines += _prevalence_summary_block(
        full_prevalence, saved_prevalence, splits, num_classes
    ).splitlines()

    lines += ["", "## Part 2 — false-positive components (baseline bias 0)", ""]
    if not component_rows:
        lines.append("No false-positive components found.")
    else:
        lines += [
            "| analysis mask | total FP area | components | mean area | median area |"
            + "".join(f" frac < {t} |" for t in area_thresholds)
            + " border fraction |",
            "|---|---:|---:|---:|---:|" + "---:|" * (len(area_thresholds) + 1),
        ]
        for analysis in ANALYSIS_MASKS:
            rows = [r for r in summary_rows if r["analysis_mask"] == analysis]
            lines.append(_pooled_component_line(analysis, rows, area_thresholds))

    lines += ["", "## Part 3 — background-bias sweep (overall group)", ""]
    lines += [
        "| background bias | object Dice | object IoU | object precision | object recall"
        " | predicted foreground px |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    zero_index = next((i for i, d in enumerate(accumulators.bias_grid) if d == 0.0), None)
    for delta_index, delta in enumerate(accumulators.bias_grid):
        if "overall" not in accumulators._confusions[delta_index]:
            continue
        confusion = accumulators.confusion(delta_index, "overall")
        obj = eval_module.object_metrics_from_3x3(confusion)
        obj_precision, obj_recall = object_precision_recall_from_3x3(confusion)
        fg_pixels = int(confusion[:, 1:].sum())
        marker = " (baseline)" if delta_index == zero_index else ""
        lines.append(
            f"| {float(delta)} | {float(obj['dice']):.4f} | {float(obj['iou']):.4f} "
            f"| {obj_precision:.4f} | {obj_recall:.4f} | {fg_pixels}{marker} |"
        )

    path.write_text("\n".join(lines) + "\n")


def _pooled_component_line(
    analysis: str, rows: list[dict[str, Any]], area_thresholds: tuple[int, ...]
) -> str:
    """Pooled component summary line for one analysis mask across images."""
    total_area = int(sum(r["total_fp_area"] for r in rows))
    n_comp = int(sum(r["n_components"] for r in rows))
    if n_comp == 0:
        cells = "".join(" 0.0000 |" for _ in area_thresholds)
        return f"| {analysis} | 0 | 0 | 0 | 0 |{cells} 0.0000 |"
    # Recompute pooled aggregates from per-image component rows is not possible
    # here (only summaries are kept per image); pool the available statistics
    # weighted by component count, and threshold areas as exact sums.
    mean_area = float(sum(r["mean_area"] * r["n_components"] for r in rows) / n_comp)
    # Pixel-weighted median is not recoverable per image; report the median of
    # per-image medians as a robust central value.
    median_area = float(np.median([r["median_area"] for r in rows if r["n_components"]]))
    frac_cells = ""
    for threshold in area_thresholds:
        below = int(sum(r[f"area_below_{threshold}"] for r in rows))
        frac = float(below) / total_area if total_area else 0.0
        frac_cells += f" {frac:.4f} |"
    border_area = int(sum(r["border_fp_area"] for r in rows))
    border_frac = float(border_area) / total_area if total_area else 0.0
    return (
        f"| {analysis} | {total_area} | {n_comp} | {mean_area:.1f} | {median_area:.1f} "
        f"|{frac_cells} {border_frac:.4f} |"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the validation diagnostics command."""
    parser = argparse.ArgumentParser(
        description="Validation diagnostics: prevalence, false-positive components, "
        "and a post-hoc background-logit-bias sweep. Read-only; never trains."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Split to analyze (default: val).",
    )
    parser.add_argument(
        "--run-dir", default=None, help="Run directory containing a best checkpoint."
    )
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint file path.")
    parser.add_argument(
        "--background-bias-grid",
        default=",".join(repr(float(v)) for v in DEFAULT_BACKGROUND_BIAS_GRID),
        help="Comma-separated post-hoc background-logit biases (default: 0,0.25,0.5,0.75,1,1.5,2).",
    )
    parser.add_argument(
        "--area-thresholds",
        default=",".join(str(v) for v in DEFAULT_AREA_THRESHOLDS),
        help="Comma-separated component area thresholds in pixels (default: 16,64,256,1024,4096).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for diagnostics outputs (default: outputs/diagnostics/).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Analyze at most this many images from the split. Smoke/debug option "
        "only; never use it for the real-data acceptance run.",
    )
    parser.add_argument(
        "--skip-saved-patch-prevalence",
        action="store_true",
        help="Skip the saved-patch prevalence analysis when the run's "
        "training_patches.npz is unavailable. The skip is recorded in summary.md.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = eval_module.load_config(config_path)

    try:
        bias_grid = parse_bias_grid(args.background_bias_grid)
        area_thresholds = parse_area_thresholds(args.area_thresholds)
        if args.max_images is not None and args.max_images <= 0:
            raise ValueError(f"--max-images must be a positive integer, got {args.max_images}")
        checkpoint_path = eval_module.resolve_checkpoint(
            config, config_path, args.run_dir, args.checkpoint
        )
        run_dir = Path(args.run_dir) if args.run_dir else checkpoint_path.parent.parent
        output_root = (
            Path(args.output_root)
            if args.output_root
            else Path(config["outputs"]["checkpoints_dir"]).parent / "diagnostics"
        )
        print(f"Using checkpoint: {checkpoint_path}")
        output_dir = run_diagnostics(
            config,
            config_path,
            split=args.split,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            bias_grid=bias_grid,
            area_thresholds=area_thresholds,
            output_root=output_root,
            max_images=args.max_images,
            skip_saved_patch_prevalence=args.skip_saved_patch_prevalence,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Diagnostics written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
