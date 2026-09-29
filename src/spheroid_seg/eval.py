"""Evaluation CLI for the spheroid segmentation model."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Cap glibc malloc arenas (default: 8 x ncores) so the streaming loop reuses
# freed blocks across images instead of growing the resident set by roughly
# one image's worth of allocator churn per iteration (measured at ~0.8 GB per
# 50 MP image on a 16-core machine). Must be set before threads allocate.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from spheroid_seg.checkpoints import resolve_checkpoint
from spheroid_seg.data.dataset import has_real_pairs, load_pair
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
from spheroid_seg.models.unet import UNet
from spheroid_seg.overlays import (
    assemble_overlay_grid,
    build_overlay_panels,
    select_overlay_samples,
)

CLASS_NAMES = ["background", "loose cell", "aggregate"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML configuration file."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_model_and_checkpoint(
    config: dict[str, Any],
    checkpoint_path: Path,
) -> tuple[UNet, dict[str, Any]]:
    """Build the model from config and restore the checkpoint.

    Args:
        config: Loaded configuration dictionary.
        checkpoint_path: Path to the Flax checkpoint file.

    Returns:
        Tuple of (model, checkpoint dictionary).

    Raises:
        ValueError: If the checkpoint is incompatible with the config.
    """
    model = UNet(
        num_classes=config["num_classes"],
        base_features=config["base_features"],
        input_channels=config["input_channels"],
        bn_momentum=config.get("bn_momentum", 0.99),
    )
    channels = 1 if config["input_channels"] == "grayscale" else 3
    patch_size = config["patch_size"]
    dummy = jnp.ones((1, patch_size, patch_size, channels), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), dummy, train=False)

    target = {
        "params": variables["params"],
        "batch_stats": variables["batch_stats"],
        "epoch": 0,
    }

    try:
        with checkpoint_path.open("rb") as f:
            ckpt = flax.serialization.from_bytes(target, f.read())
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is incompatible with the config: {exc}"
        ) from exc

    if "batch_stats" not in ckpt:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing 'batch_stats'.")

    return model, ckpt


def _make_predict_fn(apply_fn: Any) -> Any:
    """Build a JIT-compiled deterministic prediction function.

    Returns per-tile softmax probabilities (float32, ``(B, T, T, C)``) rather
    than argmax masks: proper scoring rules (Brier, log loss) are computed from
    the probabilities, and the argmax is taken per tile afterwards.
    """

    @jax.jit
    def predict(params: Any, batch_stats: Any, images: jnp.ndarray) -> jnp.ndarray:
        logits = apply_fn(
            {"params": params, "batch_stats": batch_stats},
            images,
            train=False,
            mutable=False,
        )
        return jax.nn.softmax(logits, axis=-1)

    return predict


def _tile_valid_slices(padding: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    """Per-tile ``(row_start, row_end, col_start, col_end)`` slices excluding reflect padding.

    ``extract_tiles`` reflect-pads images that are not divisible by the tile
    size; those reflected pixels are not real image pixels, so the proper
    scores must only accumulate on the valid (interior) region of each tile.
    With zero padding every slice is the full tile.
    """
    tile_size = padding["tile_size"]
    n_h, n_w = padding["n_h"], padding["n_w"]
    padded_h, padded_w = n_h * tile_size, n_w * tile_size
    row_lo, row_hi = padding["pad_top"], padded_h - padding["pad_bottom"]
    col_lo, col_hi = padding["pad_left"], padded_w - padding["pad_right"]

    slices = []
    for idx in range(n_h * n_w):
        tile_row, tile_col = divmod(idx, n_w)
        r0 = tile_row * tile_size
        c0 = tile_col * tile_size
        slices.append(
            (
                max(r0, row_lo) - r0,
                min(r0 + tile_size, row_hi) - r0,
                max(c0, col_lo) - c0,
                min(c0 + tile_size, col_hi) - c0,
            )
        )
    return slices


def _find_file_with_stem(directory: Path, stem: str) -> Path | None:
    """Return the first file in directory whose stem matches, or None."""
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem == stem:
            return path
    return None


def _resolve_split_entries(
    config: dict[str, Any],
    split: str,
    tmp_dir: Path,
) -> list[tuple[Path, Path, str]]:
    """Resolve (raw_path, mask_path, name) triples for the requested split.

    Only paths are resolved here; pixel data is streamed one image at a time by
    :func:`evaluate_split` so peak memory stays bounded by a single image
    regardless of how many images the split contains.
    """
    raw_dir = Path(config["data"]["raw_dir"])
    masks_dir = Path(config["data"]["masks_dir"])
    splits_dir = Path(config["data"]["splits_dir"])

    entries: list[tuple[Path, Path, str]] = []

    if has_real_pairs(raw_dir, masks_dir):
        names = load_split_list(splits_dir, split)
        for name in names:
            raw_path = _find_file_with_stem(raw_dir, name)
            mask_path = _find_file_with_stem(masks_dir, name)
            if raw_path is None:
                raise FileNotFoundError(f"Image '{name}' from {split}.txt not found in {raw_dir}")
            if mask_path is None:
                raise FileNotFoundError(f"Mask '{name}' from {split}.txt not found in {masks_dir}")
            entries.append((raw_path, mask_path, name))
        return entries

    # Synthetic fallback: generate the same dataset training uses and apply the
    # deterministic image-level split.
    synth_raw = tmp_dir / "raw"
    synth_masks = tmp_dir / "masks"
    n_images = config.get("synthetic_n_images", 16)
    generate_synthetic_dataset(
        synth_raw,
        synth_masks,
        n_images=n_images,
        shape=config.get("synthetic_image_shape", (512, 512)),
        seed=config["seed"],
    )
    split_names = set(synthetic_split_names(n_images, config["seed"])[split])

    for path in sorted(synth_raw.iterdir()):
        if path.suffix.lower() != ".png" or path.stem not in split_names:
            continue
        entries.append((path, synth_masks / path.name, path.stem))

    return entries


def _predict_full_image(
    predict_fn: Any,
    params: Any,
    batch_stats: Any,
    image: np.ndarray,
    mask: np.ndarray,
    tile_size: int,
    batch_size: int,
    num_classes: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Tile a full image, predict each tile, and reassemble the argmax prediction.

    Proper scoring rules (Brier, log loss) are accumulated per tile as float64
    partial sums over the valid (non-reflect-padded) region of each tile against
    the corresponding ground-truth tile; per-pixel probability maps are never
    retained beyond the current batch.

    Returns:
        Tuple of ``(argmax prediction mask, proper-score partial sums for this
        image)``.
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

    valid_slices = _tile_valid_slices(padding)
    proper_sums = empty_proper_score_sums(num_classes)
    pred_tiles: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        batch = jnp.array(tiles[start : start + batch_size], dtype=jnp.float32)
        probs = np.asarray(predict_fn(params, batch_stats, batch))
        for offset in range(len(probs)):
            idx = start + offset
            r0, r1, c0, c1 = valid_slices[idx]
            add_proper_score_sums(
                proper_sums,
                proper_score_partial_sums(
                    probs[offset, r0:r1, c0:c1],
                    mask_tiles[idx, r0:r1, c0:c1],
                    num_classes,
                ),
            )
        pred_tiles.append(np.argmax(probs, axis=-1).astype(np.uint8))

    pred_tiles_arr = np.concatenate(pred_tiles, axis=0)
    return reassemble_from_tiles(pred_tiles_arr, padding), proper_sums


def accumulate_confusion_matrix(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    num_classes: int,
) -> jnp.ndarray:
    """Compute a pixel-level confusion matrix (rows=GT, columns=prediction).

    Counts are accumulated in uint32 so that pooled counts above the float32
    mantissa limit (2**24) remain exact. The old one-hot + jnp.dot path used
    float32 internally and saturated at 2**24. uint32 is provably safe here
    because every count is non-negative and the total number of pixels in any
    evaluation run is far below 2**32.
    """
    predictions = jnp.asarray(predictions, dtype=jnp.uint32).ravel()
    targets = jnp.asarray(targets, dtype=jnp.uint32).ravel()
    flat_idx = targets * num_classes + predictions
    counts = jnp.zeros(num_classes * num_classes, dtype=jnp.uint32)
    counts = counts.at[flat_idx].add(1)
    return counts.reshape(num_classes, num_classes)


def class_metrics_from_confusion(
    confusion: jnp.ndarray,
    *,
    epsilon: float = 1e-6,
) -> dict[str, jnp.ndarray]:
    """Compute pooled per-class Dice and IoU from a confusion matrix.

    Empty-class behavior matches :mod:`spheroid_seg.metrics`: a class absent
    from both prediction and ground truth scores 1.0.
    """
    tp = jnp.diag(confusion)
    fp = jnp.sum(confusion, axis=0) - tp
    fn = jnp.sum(confusion, axis=1) - tp

    dice = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)

    absent = (tp == 0) & (fp == 0) & (fn == 0)
    dice = jnp.where(absent, 1.0, dice)
    iou = jnp.where(absent, 1.0, iou)

    return {"dice": dice, "iou": iou}


def object_confusion_from_3x3(confusion: jnp.ndarray) -> jnp.ndarray:
    """Collapse a 3-class confusion matrix into a binary bg/object matrix.

    Virtual classes are defined as ``background = (class == 0)`` and
    ``object = (class == 1) | (class == 2)``. The returned 2x2 matrix has
    rows/ground-truth and columns/prediction ordered ``[background, object]``.

    The summation stays in uint32 so it reuses the exact-integer accumulation
    path; no new float32 count accumulation is introduced.
    """
    bg_bg = confusion[0, 0]
    bg_obj = confusion[0, 1] + confusion[0, 2]
    obj_bg = confusion[1, 0] + confusion[2, 0]
    obj_obj = confusion[1, 1] + confusion[1, 2] + confusion[2, 1] + confusion[2, 2]
    return jnp.array(
        [[bg_bg, bg_obj], [obj_bg, obj_obj]],
        dtype=jnp.uint32,
    )


def object_metrics_from_3x3(
    confusion: jnp.ndarray,
    *,
    epsilon: float = 1e-6,
) -> dict[str, jnp.ndarray]:
    """Compute the virtual object-class Dice and IoU from a 3-class confusion matrix.

    Args:
        confusion: 3x3 confusion matrix with class order
            ``[background, loose cell, aggregate]``.
        epsilon: Small constant for numerical stability.

    Returns:
        Dictionary with scalar ``dice`` and ``iou`` for the object virtual class.
    """
    obj_conf = object_confusion_from_3x3(confusion)
    pooled = class_metrics_from_confusion(obj_conf, epsilon=epsilon)
    return {"dice": pooled["dice"][1], "iou": pooled["iou"][1]}


def _dice_iou_from_confusion(
    confusion: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class Dice and IoU derived from a confusion matrix.

    Bit-identical to :func:`spheroid_seg.metrics.dice_score` and
    :func:`spheroid_seg.metrics.iou_score` on the same pixels: the underlying
    counts (intersection, prediction area, target area) are exact uint32
    integers either way, and the closing float32 expressions are the same.
    Deriving them from the already-materialized per-image confusion avoids
    building full-image one-hot arrays (~1.5 GB of transient buffers per
    50 MP image).
    """
    confusion = jnp.asarray(confusion)
    tp = jnp.diag(confusion)
    prediction_area = jnp.sum(confusion, axis=0)
    target_area = jnp.sum(confusion, axis=1)
    dice = (2.0 * tp + epsilon) / (prediction_area + target_area + epsilon)
    iou = (tp + epsilon) / (prediction_area + target_area - tp + epsilon)
    absent = (prediction_area == 0) & (target_area == 0)
    return (
        np.asarray(jnp.where(absent, 1.0, dice)),
        np.asarray(jnp.where(absent, 1.0, iou)),
    )


def group_by_magnification(
    names: list[str],
    values: list[Any],
) -> dict[str, list[Any]]:
    """Group values by the magnification parsed from each name."""
    groups: dict[str, list[Any]] = {}
    for name, value in zip(names, values, strict=False):
        mag = parse_magnification(name)
        groups.setdefault(mag, []).append(value)
    return groups


def _score_cell(value: Any) -> Any:
    """CSV cell for a proper score: empty for NaN/undefined, the float otherwise."""
    v = float(value)
    return "" if math.isnan(v) else v


def _write_outputs(
    output_dir: Path,
    metrics: dict[str, Any],
    overlay_samples: list[dict[str, Any]],
    n_images: int,
    config: dict[str, Any],
) -> None:
    """Write metrics.json, metrics.csv, confusion matrices, and overlay grid.

    Writes the original 3x3 ``confusion_matrix.csv`` plus an optional
    ``confusion_matrix_object.csv`` (2x2 virtual bg/object) derived from it.
    ``metrics.csv`` has one block per group (overall + each magnification):
    one row per model class with per-class one-vs-rest Brier and conditional
    log loss, an ``object`` row with the proper-score cells left empty (the
    object class is a post-hoc merge of two softmax classes, not a model
    output), and an ``all`` row with the pooled multiclass Brier and overall
    log loss over all pixels and classes. ``overlay_samples`` carries pixel
    data only for the images selected as overlay panels, pre-rendered at panel
    width; selection is re-applied here to fix the row order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    def _class_rows(group_name: str, group: dict[str, Any]) -> list[list[Any]]:
        rows = [
            [
                group_name,
                class_name,
                group["dice"][idx],
                group["iou"][idx],
                _score_cell(group["brier"][idx]),
                _score_cell(group["log_loss"][idx]),
                group["n_images"],
            ]
            for idx, class_name in enumerate(CLASS_NAMES[: config["num_classes"]])
        ]
        rows.append(
            [
                group_name,
                "object",
                group["object_dice"],
                group["object_iou"],
                "",
                "",
                group["n_images"],
            ]
        )
        rows.append(
            [
                group_name,
                "all",
                "",
                "",
                _score_cell(group["brier_all"]),
                _score_cell(group["log_loss_all"]),
                group["n_images"],
            ]
        )
        return rows

    num_classes = config["num_classes"]
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "class", "dice", "iou", "brier", "log_loss", "n_images"])
        overall_group = dict(metrics["overall"])
        overall_group["n_images"] = n_images
        writer.writerows(_class_rows("overall", overall_group))
        for mag, group in metrics["per_magnification"].items():
            writer.writerows(_class_rows(mag, group))

    with (output_dir / "confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + CLASS_NAMES[:num_classes])
        for idx, class_name in enumerate(CLASS_NAMES[:num_classes]):
            writer.writerow([class_name] + metrics["confusion_matrix"][idx])

    with (output_dir / "confusion_matrix_object.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        obj_names = ["background", "object"]
        writer.writerow([""] + obj_names)
        for idx, class_name in enumerate(obj_names):
            writer.writerow([class_name] + metrics["confusion_matrix_object"][idx])

    eval_config = config.get("eval", {})
    num_overlay_samples = eval_config.get("num_overlay_samples", 8)
    panel_width = eval_config.get("overlay_panel_width", 384)

    selected = select_overlay_samples(overlay_samples, num_overlay_samples)
    if selected:
        import cv2

        grid = assemble_overlay_grid(selected, panel_width)
        cv2.imwrite(str(output_dir / "overlays_grid.png"), grid)


def _float_image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8 grayscale."""
    if image.ndim == 3 and image.shape[2] == 3:
        # RGB: convert to grayscale for the overlay raw panel.
        image = np.mean(image, axis=-1)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def _make_output_dir(config: dict[str, Any], config_path: Path | str) -> Path:
    """Create a unique evaluation output directory."""
    checkpoints_dir = Path(config["outputs"]["checkpoints_dir"])
    evals_dir = checkpoints_dir.parent / "evals"
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"{Path(config_path).stem}_{timestamp}"
    output_dir = evals_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _fmt_score(value: Any) -> str:
    """Right-aligned score cell for the stdout summary (NaN prints as 'nan')."""
    return f"{float(value):>8.4f}"


def _print_summary(metrics: dict[str, Any]) -> None:
    """Print a compact summary table to stdout."""
    print("\nEvaluation summary")
    print("-" * 76)
    print(f"{'Group':<12} {'Class':<14} {'Dice':>8} {'IoU':>8} {'Brier':>8} {'LogLoss':>8}")
    print("-" * 76)

    def _rows(group_name: str, group: dict[str, Any]) -> list[str]:
        lines = [
            f"{group_name:<12} {class_name:<14} "
            f"{group['dice'][idx]:>8.4f} {group['iou'][idx]:>8.4f} "
            f"{_fmt_score(group['brier'][idx])} {_fmt_score(group['log_loss'][idx])}"
            for idx, class_name in enumerate(metrics["class_names"])
        ]
        # object: proper scores are undefined (post-hoc merge, not a softmax output).
        lines.append(
            f"{group_name:<12} {'object':<14} "
            f"{group['object_dice']:>8.4f} {group['object_iou']:>8.4f} "
            f"{'':>8} {'':>8}"
        )
        lines.append(
            f"{group_name:<12} {'all':<14} "
            f"{'':>8} {'':>8} "
            f"{_fmt_score(group['brier_all'])} {_fmt_score(group['log_loss_all'])}"
        )
        return lines

    for line in _rows("overall", metrics["overall"]):
        print(line)
    for mag, group in sorted(metrics["per_magnification"].items()):
        for line in _rows(mag, group):
            print(line)
    print("-" * 76)


def evaluate_split(
    config: dict[str, Any],
    config_path: Path | str,
    split: str,
    checkpoint_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Run evaluation for a split and write all outputs.

    Images are streamed one at a time (load -> predict -> accumulate ->
    release); across images only small aggregated state is retained: pooled
    confusion counts, per-image scalar metrics, and the panel-resolution
    overlay renderings of the few images selected for the grid. Peak memory
    therefore stays bounded by one image plus a constant overlay budget,
    independent of split size.

    Outputs are written to a unique ``outputs/evals/<config>_<timestamp>/``
    directory: ``metrics.json``, ``metrics.csv`` (per-class Dice/IoU plus Brier
    and log-loss proper scoring rules), ``confusion_matrix.csv``,
    ``confusion_matrix_object.csv``, and ``overlays_grid.png``.

    Returns:
        Tuple of (output directory, metrics dictionary).
    """
    model, ckpt = load_model_and_checkpoint(config, checkpoint_path)
    predict_fn = _make_predict_fn(model.apply)

    num_classes = config["num_classes"]
    eval_config = config.get("eval", {})
    eval_batch_size = eval_config.get("batch_size", config["batch_size"])
    num_overlay_samples = eval_config.get("num_overlay_samples", 8)
    panel_width = eval_config.get("overlay_panel_width", 384)

    with tempfile.TemporaryDirectory() as tmp:
        entries = _resolve_split_entries(config, split, Path(tmp))
        if not entries:
            raise ValueError(f"No images found for split '{split}'.")

        # Overlay selection is a pure function of (name, magnification), so
        # decide it on lightweight metadata and keep pixel data only for the
        # images that will actually be rendered in the grid.
        names = [name for _, _, name in entries]
        mags = [parse_magnification(name) for name in names]
        selected_names = {
            sample["name"]
            for sample in select_overlay_samples(
                [{"name": n, "magnification": m} for n, m in zip(names, mags, strict=True)],
                num_overlay_samples,
            )
        }

        global_confusion = np.zeros((num_classes, num_classes), dtype=np.uint32)
        overall_proper_sums = empty_proper_score_sums(num_classes)
        group_proper_sums: dict[str, dict[str, np.ndarray]] = {}
        group_confusions: dict[str, np.ndarray] = {}
        group_dices: dict[str, list[np.ndarray]] = {}
        group_ious: dict[str, list[np.ndarray]] = {}
        group_object_dices: dict[str, list[float]] = {}
        group_object_ious: dict[str, list[float]] = {}
        per_image: list[dict[str, Any]] = []
        overlay_samples: list[dict[str, Any]] = []

        input_channels = config["input_channels"]
        class_mapping = config["class_mapping"]
        for raw_path, mask_path, name in entries:
            image, mask = load_pair(
                raw_path,
                mask_path,
                input_channels=input_channels,
                class_mapping=class_mapping,
            )
            pred, proper_sums = _predict_full_image(
                predict_fn,
                ckpt["params"],
                ckpt["batch_stats"],
                image,
                mask,
                tile_size=config["patch_size"],
                batch_size=eval_batch_size,
                num_classes=num_classes,
            )

            # Per-image confusion first; every pooled matrix below is the exact
            # integer sum of per-image matrices (uint32 accumulation, as
            # documented in accumulate_confusion_matrix). Proper scores are the
            # exact float64 sum of per-tile partial sums, folded into the
            # overall and per-magnification accumulators.
            conf = np.asarray(accumulate_confusion_matrix(pred, mask, num_classes))
            pooled = class_metrics_from_confusion(conf)
            obj = object_metrics_from_3x3(conf)
            mag = parse_magnification(name)

            global_confusion += conf
            group_confusions[mag] = group_confusions.get(mag, np.zeros_like(conf)) + conf
            add_proper_score_sums(overall_proper_sums, proper_sums)
            add_proper_score_sums(
                group_proper_sums.setdefault(mag, empty_proper_score_sums(num_classes)),
                proper_sums,
            )
            dice, iou = _dice_iou_from_confusion(conf)
            group_dices.setdefault(mag, []).append(dice)
            group_ious.setdefault(mag, []).append(iou)
            group_object_dices.setdefault(mag, []).append(float(obj["dice"]))
            group_object_ious.setdefault(mag, []).append(float(obj["iou"]))
            per_image.append(
                {
                    "name": name,
                    "magnification": mag,
                    "dice": np.asarray(pooled["dice"]).tolist(),
                    "iou": np.asarray(pooled["iou"]).tolist(),
                    "object_dice": float(obj["dice"]),
                    "object_iou": float(obj["iou"]),
                }
            )
            if name in selected_names:
                # Render the small grid panels now, while the full-resolution
                # arrays are still in memory; only the resized panels are kept.
                overlay_sample = {
                    "raw": _float_image_to_uint8(image),
                    "gt": mask.astype(np.uint8),
                    "pred": pred.astype(np.uint8),
                }
                overlay_samples.append(
                    {
                        "name": name,
                        "magnification": mag,
                        "panels": build_overlay_panels(overlay_sample, panel_width),
                    }
                )

        overall = class_metrics_from_confusion(global_confusion)
        overall_object = object_metrics_from_3x3(global_confusion)
        overall_scores = finalize_proper_scores(overall_proper_sums, num_classes)

        per_magnification: dict[str, Any] = {}
        for mag in sorted(group_confusions.keys()):
            conf = group_confusions[mag]
            pooled = class_metrics_from_confusion(conf)
            obj = object_metrics_from_3x3(conf)
            group_scores = finalize_proper_scores(group_proper_sums[mag], num_classes)
            per_magnification[mag] = {
                "dice": np.asarray(pooled["dice"]).tolist(),
                "iou": np.asarray(pooled["iou"]).tolist(),
                "object_dice": float(obj["dice"]),
                "object_iou": float(obj["iou"]),
                "brier": np.asarray(group_scores["brier_per_class"]).tolist(),
                "brier_all": float(group_scores["brier"]),
                "log_loss": np.asarray(group_scores["log_loss_per_class"]).tolist(),
                "log_loss_all": float(group_scores["log_loss"]),
                "per_image": {
                    "mean_dice": np.mean(group_dices[mag], axis=0).tolist(),
                    "std_dice": np.std(group_dices[mag], axis=0).tolist(),
                    "mean_iou": np.mean(group_ious[mag], axis=0).tolist(),
                    "std_iou": np.std(group_ious[mag], axis=0).tolist(),
                    "mean_object_dice": float(np.mean(group_object_dices[mag])),
                    "std_object_dice": float(np.std(group_object_dices[mag])),
                    "mean_object_iou": float(np.mean(group_object_ious[mag])),
                    "std_object_iou": float(np.std(group_object_ious[mag])),
                },
                "n_images": len(group_dices[mag]),
            }

        metrics = {
            "overall": {
                "dice": np.asarray(overall["dice"]).tolist(),
                "iou": np.asarray(overall["iou"]).tolist(),
                "object_dice": float(overall_object["dice"]),
                "object_iou": float(overall_object["iou"]),
                "brier": np.asarray(overall_scores["brier_per_class"]).tolist(),
                "brier_all": float(overall_scores["brier"]),
                "log_loss": np.asarray(overall_scores["log_loss_per_class"]).tolist(),
                "log_loss_all": float(overall_scores["log_loss"]),
            },
            "per_magnification": per_magnification,
            "per_image": per_image,
            "confusion_matrix": global_confusion.tolist(),
            "confusion_matrix_object": np.asarray(
                object_confusion_from_3x3(global_confusion)
            ).tolist(),
            "class_names": CLASS_NAMES,
        }

        output_dir = _make_output_dir(config, config_path)
        _write_outputs(output_dir, metrics, overlay_samples, len(entries), config)
        _print_summary(metrics)

    return output_dir, metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate the spheroid segmentation model.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Split to evaluate (default: val).",
    )
    parser.add_argument(
        "--run-dir", default=None, help="Run directory containing a best checkpoint."
    )
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint file path.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)

    try:
        checkpoint_path = resolve_checkpoint(config, config_path, args.run_dir, args.checkpoint)
        print(f"Using checkpoint: {checkpoint_path}")
        evaluate_split(config, config_path, args.split, checkpoint_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
