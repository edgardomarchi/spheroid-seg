"""Evaluation CLI for the spheroid segmentation model."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

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
from spheroid_seg.metrics import dice_score, iou_score
from spheroid_seg.models.unet import UNet
from spheroid_seg.overlays import build_overlay_grid, select_overlay_samples

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
    """Build a JIT-compiled deterministic prediction function."""

    @jax.jit
    def predict(params: Any, batch_stats: Any, images: jnp.ndarray) -> jnp.ndarray:
        logits = apply_fn(
            {"params": params, "batch_stats": batch_stats},
            images,
            train=False,
            mutable=False,
        )
        return jnp.argmax(logits, axis=-1)

    return predict


def _find_file_with_stem(directory: Path, stem: str) -> Path | None:
    """Return the first file in directory whose stem matches, or None."""
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem == stem:
            return path
    return None


def _load_split_samples(
    config: dict[str, Any],
    split: str,
    tmp_dir: Path,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Load all (image, mask, name) tuples for the requested split."""
    raw_dir = Path(config["data"]["raw_dir"])
    masks_dir = Path(config["data"]["masks_dir"])
    splits_dir = Path(config["data"]["splits_dir"])

    input_channels = config["input_channels"]
    class_mapping = config["class_mapping"]
    samples: list[tuple[np.ndarray, np.ndarray, str]] = []

    if has_real_pairs(raw_dir, masks_dir):
        names = load_split_list(splits_dir, split)
        for name in names:
            raw_path = _find_file_with_stem(raw_dir, name)
            mask_path = _find_file_with_stem(masks_dir, name)
            if raw_path is None:
                raise FileNotFoundError(f"Image '{name}' from {split}.txt not found in {raw_dir}")
            if mask_path is None:
                raise FileNotFoundError(f"Mask '{name}' from {split}.txt not found in {masks_dir}")
            image, mask = load_pair(
                raw_path,
                mask_path,
                input_channels=input_channels,
                class_mapping=class_mapping,
            )
            samples.append((image, mask, name))
        return samples

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
        mask_path = synth_masks / path.name
        image, mask = load_pair(
            path,
            mask_path,
            input_channels=input_channels,
            class_mapping=class_mapping,
        )
        samples.append((image, mask, path.stem))

    return samples


def _predict_full_image(
    predict_fn: Any,
    params: Any,
    batch_stats: Any,
    image: np.ndarray,
    tile_size: int,
    batch_size: int,
) -> np.ndarray:
    """Tile a full image, predict each tile, and reassemble the prediction."""
    tiles, padding = extract_tiles(image, tile_size)
    if tiles.ndim == 3:
        tiles = tiles[..., np.newaxis]

    pred_tiles: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        batch = jnp.array(tiles[start : start + batch_size], dtype=jnp.float32)
        preds = predict_fn(params, batch_stats, batch)
        pred_tiles.append(np.asarray(preds))

    pred_tiles = np.concatenate(pred_tiles, axis=0)
    return reassemble_from_tiles(pred_tiles, padding)


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


def _compute_metrics(
    samples: list[tuple[np.ndarray, np.ndarray, str]],
    predictions: list[np.ndarray],
    num_classes: int,
) -> dict[str, Any]:
    """Compute overall, per-magnification, and per-image metrics."""
    names = [name for _, _, name in samples]
    masks = [mask for _, mask, _ in samples]

    all_pred = jnp.concatenate([jnp.asarray(p).ravel() for p in predictions])
    all_target = jnp.concatenate([jnp.asarray(m).ravel() for m in masks])
    global_confusion = accumulate_confusion_matrix(all_pred, all_target, num_classes)
    overall = class_metrics_from_confusion(global_confusion)
    overall_object = object_metrics_from_3x3(global_confusion)

    per_image: list[dict[str, Any]] = []
    for (_, mask, name), pred in zip(samples, predictions, strict=False):
        conf = accumulate_confusion_matrix(pred, mask, num_classes)
        pooled = class_metrics_from_confusion(conf)
        obj = object_metrics_from_3x3(conf)
        per_image.append(
            {
                "name": name,
                "magnification": parse_magnification(name),
                "dice": np.asarray(pooled["dice"]).tolist(),
                "iou": np.asarray(pooled["iou"]).tolist(),
                "object_dice": float(obj["dice"]),
                "object_iou": float(obj["iou"]),
            }
        )

    grouped_preds = group_by_magnification(names, predictions)
    grouped_masks = group_by_magnification(names, masks)

    per_magnification: dict[str, Any] = {}
    for mag in sorted(grouped_preds.keys()):
        group_pred = jnp.concatenate([jnp.asarray(p).ravel() for p in grouped_preds[mag]])
        group_target = jnp.concatenate([jnp.asarray(m).ravel() for m in grouped_masks[mag]])
        conf = accumulate_confusion_matrix(group_pred, group_target, num_classes)
        pooled = class_metrics_from_confusion(conf)
        obj = object_metrics_from_3x3(conf)

        group_dices = [
            np.asarray(dice_score(p, m, num_classes))
            for p, m in zip(grouped_preds[mag], grouped_masks[mag], strict=False)
        ]
        group_ious = [
            np.asarray(iou_score(p, m, num_classes))
            for p, m in zip(grouped_preds[mag], grouped_masks[mag], strict=False)
        ]
        group_object_dices = []
        group_object_ious = []
        for p, m in zip(grouped_preds[mag], grouped_masks[mag], strict=False):
            im_conf = accumulate_confusion_matrix(p, m, num_classes)
            im_obj = object_metrics_from_3x3(im_conf)
            group_object_dices.append(float(im_obj["dice"]))
            group_object_ious.append(float(im_obj["iou"]))

        per_magnification[mag] = {
            "dice": np.asarray(pooled["dice"]).tolist(),
            "iou": np.asarray(pooled["iou"]).tolist(),
            "object_dice": float(obj["dice"]),
            "object_iou": float(obj["iou"]),
            "per_image": {
                "mean_dice": np.mean(group_dices, axis=0).tolist(),
                "std_dice": np.std(group_dices, axis=0).tolist(),
                "mean_iou": np.mean(group_ious, axis=0).tolist(),
                "std_iou": np.std(group_ious, axis=0).tolist(),
                "mean_object_dice": float(np.mean(group_object_dices)),
                "std_object_dice": float(np.std(group_object_dices)),
                "mean_object_iou": float(np.mean(group_object_ious)),
                "std_object_iou": float(np.std(group_object_ious)),
            },
            "n_images": len(grouped_preds[mag]),
        }

    return {
        "overall": {
            "dice": np.asarray(overall["dice"]).tolist(),
            "iou": np.asarray(overall["iou"]).tolist(),
            "object_dice": float(overall_object["dice"]),
            "object_iou": float(overall_object["iou"]),
        },
        "per_magnification": per_magnification,
        "per_image": per_image,
        "confusion_matrix": np.asarray(global_confusion).tolist(),
        "confusion_matrix_object": np.asarray(object_confusion_from_3x3(global_confusion)).tolist(),
        "class_names": CLASS_NAMES,
    }


def _write_outputs(
    output_dir: Path,
    metrics: dict[str, Any],
    samples: list[tuple[np.ndarray, np.ndarray, str]],
    predictions: list[np.ndarray],
    config: dict[str, Any],
) -> None:
    """Write metrics.json, metrics.csv, confusion matrices, and overlay grid.

    Writes the original 3x3 ``confusion_matrix.csv`` plus an optional
    ``confusion_matrix_object.csv`` (2x2 virtual bg/object) derived from it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    num_classes = config["num_classes"]
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "class", "dice", "iou", "n_images"])
        for idx, class_name in enumerate(CLASS_NAMES[:num_classes]):
            writer.writerow(
                [
                    "overall",
                    class_name,
                    metrics["overall"]["dice"][idx],
                    metrics["overall"]["iou"][idx],
                    len(samples),
                ]
            )
        writer.writerow(
            [
                "overall",
                "object",
                metrics["overall"]["object_dice"],
                metrics["overall"]["object_iou"],
                len(samples),
            ]
        )
        for mag, group in metrics["per_magnification"].items():
            for idx, class_name in enumerate(CLASS_NAMES[:num_classes]):
                writer.writerow(
                    [
                        mag,
                        class_name,
                        group["dice"][idx],
                        group["iou"][idx],
                        group["n_images"],
                    ]
                )
            writer.writerow(
                [
                    mag,
                    "object",
                    group["object_dice"],
                    group["object_iou"],
                    group["n_images"],
                ]
            )

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

    overlay_samples = [
        {
            "name": name,
            "magnification": parse_magnification(name),
            "raw": _float_image_to_uint8(image),
            "gt": mask.astype(np.uint8),
            "pred": pred.astype(np.uint8),
        }
        for (_, mask, name), pred, (image, _, _) in zip(samples, predictions, samples, strict=False)
    ]
    selected = select_overlay_samples(overlay_samples, num_overlay_samples)
    if selected:
        import cv2

        grid = build_overlay_grid(selected, panel_width)
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


def _print_summary(metrics: dict[str, Any]) -> None:
    """Print a compact summary table to stdout."""
    print("\nEvaluation summary")
    print("-" * 60)
    print(f"{'Group':<12} {'Class':<14} {'Dice':>8} {'IoU':>8}")
    print("-" * 60)
    for idx, class_name in enumerate(metrics["class_names"]):
        print(
            f"{'overall':<12} {class_name:<14} "
            f"{metrics['overall']['dice'][idx]:>8.4f} {metrics['overall']['iou'][idx]:>8.4f}"
        )
    print(
        f"{'overall':<12} {'object':<14} "
        f"{metrics['overall']['object_dice']:>8.4f} {metrics['overall']['object_iou']:>8.4f}"
    )
    for mag, group in sorted(metrics["per_magnification"].items()):
        for idx, class_name in enumerate(metrics["class_names"]):
            print(
                f"{mag:<12} {class_name:<14} {group['dice'][idx]:>8.4f} {group['iou'][idx]:>8.4f}"
            )
        print(f"{mag:<12} {'object':<14} {group['object_dice']:>8.4f} {group['object_iou']:>8.4f}")
    print("-" * 60)


def evaluate_split(
    config: dict[str, Any],
    config_path: Path | str,
    split: str,
    checkpoint_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Run evaluation for a split and write all outputs.

    Outputs are written to a unique ``outputs/evals/<config>_<timestamp>/``
    directory: ``metrics.json``, ``metrics.csv``, ``confusion_matrix.csv``,
    ``confusion_matrix_object.csv``, and ``overlays_grid.png``.

    Returns:
        Tuple of (output directory, metrics dictionary).
    """
    model, ckpt = load_model_and_checkpoint(config, checkpoint_path)
    predict_fn = _make_predict_fn(model.apply)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        samples = _load_split_samples(config, split, tmp_dir)
        if not samples:
            raise ValueError(f"No images found for split '{split}'.")

        predictions: list[np.ndarray] = []
        eval_batch_size = config.get("eval", {}).get("batch_size", config["batch_size"])
        for image, _, _ in samples:
            pred = _predict_full_image(
                predict_fn,
                ckpt["params"],
                ckpt["batch_stats"],
                image,
                tile_size=config["patch_size"],
                batch_size=eval_batch_size,
            )
            predictions.append(np.asarray(pred))

        metrics = _compute_metrics(samples, predictions, config["num_classes"])
        output_dir = _make_output_dir(config, config_path)
        _write_outputs(output_dir, metrics, samples, predictions, config)
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
