"""Inference CLI for full-image semantic segmentation via patch stitching."""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Any

import cv2
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from spheroid_seg.checkpoints import resolve_checkpoint
from spheroid_seg.data.dataset import IMAGE_EXTENSIONS, _to_grayscale, _to_rgb, normalize_percentile
from spheroid_seg.data.stitching import compute_stride, extract_overlapping_tiles, stitch_logits
from spheroid_seg.models.unet import UNet
from spheroid_seg.overlays import CLASS_COLORMAP


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


def _make_logit_predict_fn(apply_fn: Any) -> Any:
    """Build a JIT-compiled deterministic logit prediction function."""

    @jax.jit
    def predict(params: Any, batch_stats: Any, images: jnp.ndarray) -> jnp.ndarray:
        return apply_fn(
            {"params": params, "batch_stats": batch_stats},
            images,
            train=False,
            mutable=False,
        )

    return predict


def _load_input_images(input_path: Path) -> list[Path]:
    """Resolve --input into a list of supported image paths.

    Unsupported files inside a directory are skipped with a warning to stderr.

    Raises:
        FileNotFoundError: If the input path does not exist.
        ValueError: If no supported image files are found in a directory.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]

    images: list[Path] = []
    for p in input_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
        else:
            print(f"Warning: skipping unsupported file {p.name}", file=sys.stderr)
    images.sort(key=lambda p: p.name)

    if not images:
        raise ValueError(f"No supported image files found in {input_path}")

    return images


def _preprocess_image(image_path: Path, input_channels: str) -> np.ndarray:
    """Read and normalize a raw image for inference.

    Args:
        image_path: Path to the raw image.
        input_channels: "grayscale" or "rgb".

    Returns:
        Normalized float32 image, 2D for grayscale or HxWx3 for RGB.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = normalize_percentile(image)

    if input_channels == "grayscale":
        image = _to_grayscale(image)
    elif input_channels == "rgb":
        image = _to_rgb(image)
    else:
        raise ValueError(f"input_channels must be 'rgb' or 'grayscale', got {input_channels}")

    return image.astype(np.float32)


def _predict_stitched(
    predict_fn: Any,
    params: Any,
    batch_stats: Any,
    image: np.ndarray,
    tile_size: int,
    overlap: float,
    batch_size: int,
    num_classes: int,
) -> np.ndarray:
    """Predict a full image via overlapping tiles with logit averaging."""
    stride = compute_stride(tile_size, overlap)
    tiles, origins, padding = extract_overlapping_tiles(image, tile_size, stride)

    if tiles.ndim == 3:
        tiles = tiles[..., np.newaxis]

    logit_tiles: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        batch = jnp.array(tiles[start : start + batch_size], dtype=jnp.float32)
        logits = predict_fn(params, batch_stats, batch)
        logit_tiles.append(np.asarray(logits))

    all_logits = np.concatenate(logit_tiles, axis=0)
    return stitch_logits(all_logits, origins, padding, num_classes)


def _build_prediction_overlay(raw: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Create a quick-look overlay: raw image tinted by predicted class colors."""
    if raw.ndim == 2:
        raw_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    else:
        raw_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

    colored = np.zeros_like(raw_bgr)
    for class_id, color in CLASS_COLORMAP.items():
        colored[prediction == class_id] = color

    overlay = cv2.addWeighted(raw_bgr, 0.5, colored, 0.5, 0)
    return overlay


def _make_output_dir(config: dict[str, Any], config_path: Path | str) -> Path:
    """Create a unique inference output directory."""
    checkpoints_dir = Path(config["outputs"]["checkpoints_dir"])
    infer_dir = checkpoints_dir.parent / "infer"
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"{Path(config_path).stem}_{timestamp}"
    output_dir = infer_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_inference(
    config: dict[str, Any],
    config_path: Path | str,
    input_path: Path,
    checkpoint_path: Path,
) -> tuple[Path, list[Path]]:
    """Run inference on all input images and write masks + overlays.

    Returns:
        Tuple of (output directory, list of written mask paths).
    """
    model, ckpt = load_model_and_checkpoint(config, checkpoint_path)
    predict_fn = _make_logit_predict_fn(model.apply)

    infer_config = config.get("infer", {})
    overlap = infer_config.get("overlap", 0.15)
    batch_size = infer_config.get("batch_size", config["batch_size"])

    image_paths = _load_input_images(input_path)
    output_dir = _make_output_dir(config, config_path)
    pred_dir = output_dir / "predictions"
    overlay_dir = output_dir / "overlays"
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for image_path in image_paths:
        image = _preprocess_image(image_path, config["input_channels"])
        prediction = _predict_stitched(
            predict_fn,
            ckpt["params"],
            ckpt["batch_stats"],
            image,
            tile_size=config["patch_size"],
            overlap=overlap,
            batch_size=batch_size,
            num_classes=config["num_classes"],
        )

        mask_path = pred_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(mask_path), prediction.astype(np.uint8))
        written.append(mask_path)

        raw_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        overlay = _build_prediction_overlay(raw_uint8, prediction)
        cv2.imwrite(str(overlay_dir / f"{image_path.stem}.png"), overlay)

    return output_dir, written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run full-image inference with the spheroid segmentation model.",
    )
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single image file or a directory of images.",
    )
    parser.add_argument(
        "--run-dir", default=None, help="Run directory containing a best checkpoint."
    )
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint file path.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)
    input_path = Path(args.input)

    try:
        checkpoint_path = resolve_checkpoint(config, config_path, args.run_dir, args.checkpoint)
        print(f"Using checkpoint: {checkpoint_path}")

        output_dir, written = run_inference(config, config_path, input_path, checkpoint_path)
        print("\nInference summary")
        print(f"  Images processed: {len(written)}")
        print(f"  Output directory: {output_dir}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
