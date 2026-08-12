"""Visualize a grid of augmented image/mask patches for QC.

If no real raw/mask pairs are present, the script deterministically generates
synthetic fixtures, augments them, and saves the resulting grid to
``outputs/debug/augmented_batches.png``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from spheroid_seg.data.augment import apply_augmentation, build_augmentation
from spheroid_seg.data.dataset import SpheroidDataset
from spheroid_seg.data.patching import extract_patches


def _get_pyplot() -> Any:
    """Import matplotlib.pyplot lazily and fail with a helpful message."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires matplotlib. "
            'Install it with: pip install "spheroid-seg[viz]" (or: pip install matplotlib). '
            "For uv development environments, use: uv sync --all-groups"
        ) from exc
    return plt


def _synthetic_fixture_dir(tmp_dir: Path, n_images: int = 4) -> tuple[Path, Path]:
    """Generate deterministic synthetic raw/mask pairs for visualization."""
    raw_dir = tmp_dir / "raw"
    masks_dir = tmp_dir / "masks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for idx in range(n_images):
        shape = (512, 512)
        raw = rng.integers(0, 256, size=shape, dtype=np.uint8)
        mask = np.zeros(shape, dtype=np.uint8)
        # Draw a few object blobs.
        for _ in range(5):
            cy, cx = rng.integers(64, shape[0] - 64, size=2)
            radius = rng.integers(20, 60)
            value = rng.choice([1, 2, 3])
            cv2.circle(mask, (int(cx), int(cy)), int(radius), int(value), -1)

        cv2.imwrite(str(raw_dir / f"synth_{idx:02d}.png"), raw)
        cv2.imwrite(str(masks_dir / f"synth_{idx:02d}.png"), mask)

    return raw_dir, masks_dir


def _load_config(config_path: Path) -> dict:
    """Load the YAML configuration file."""
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def _class_stats(mask: np.ndarray) -> dict[int, int]:
    """Return pixel counts per class ID."""
    unique, counts = np.unique(mask, return_counts=True)
    return {int(cls): int(cnt) for cls, cnt in zip(unique, counts, strict=False)}


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map class IDs to RGB colors for overlay."""
    colors = np.array(
        [
            [0, 0, 0],  # background: black
            [255, 0, 0],  # loose cell: red
            [0, 255, 0],  # aggregate: green
            [0, 0, 255],  # organoid (pre-merge): blue
        ],
        dtype=np.uint8,
    )
    return colors[np.clip(mask, 0, 3)]


def _overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a colorized mask on a grayscale image."""
    image_u8 = (image * 255).clip(0, 255).astype(np.uint8)
    image_rgb = np.stack([image_u8] * 3, axis=-1) if image_u8.ndim == 2 else image_u8

    colored = _colorize_mask(mask)
    blended = (alpha * colored + (1 - alpha) * image_rgb).astype(np.uint8)
    return blended


def main(argv: list[str] | None = None) -> int:
    """Generate augmented-batch visualization and print class statistics."""
    parser = argparse.ArgumentParser(description="Visualize augmented image/mask patches.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/debug/augmented_batches.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Number of augmented patches to visualize.",
    )
    args = parser.parse_args(argv)

    config = _load_config(Path(args.config))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(config["data"]["raw_dir"])
    masks_dir = Path(config["data"]["masks_dir"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset = SpheroidDataset(
            raw_dir if any(raw_dir.glob("*")) else _synthetic_fixture_dir(tmp_path)[0],
            masks_dir if any(masks_dir.glob("*")) else _synthetic_fixture_dir(tmp_path)[1],
            input_channels=config["input_channels"],
            class_mapping=config["class_mapping"],
        )

        if len(dataset) == 0:
            print("No paired raw/mask images found; generating synthetic fixtures.")
            synth_raw, synth_masks = _synthetic_fixture_dir(tmp_path)
            dataset = SpheroidDataset(
                synth_raw,
                synth_masks,
                input_channels=config["input_channels"],
                class_mapping=config["class_mapping"],
            )

        augment = build_augmentation(config["augment"], seed=config["seed"])

        samples: list[tuple[np.ndarray, np.ndarray]] = []
        rng = np.random.default_rng(config["seed"])
        while len(samples) < args.n_samples:
            idx = rng.integers(0, len(dataset))
            image, mask, _ = dataset[idx]
            patches_img, patches_mask = extract_patches(
                image,
                mask,
                patch_size=config["patch_size"],
                min_object_fraction=config["min_object_fraction"],
                object_patch_ratio=config["object_patch_ratio"],
                patches_per_image=max(1, args.n_samples // len(dataset) + 1),
                seed=int(rng.integers(0, 2**31)),
            )
            for p_img, p_mask in zip(patches_img, patches_mask, strict=False):
                aug_img, aug_mask = apply_augmentation(augment, p_img, p_mask)
                samples.append((aug_img, aug_mask))
                if len(samples) >= args.n_samples:
                    break

    n_cols = 4
    n_rows = (args.n_samples + n_cols - 1) // n_cols
    plt = _get_pyplot()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    print("\nPer-sample class stats:")
    for idx, (image, mask) in enumerate(samples[: args.n_samples]):
        stats = _class_stats(mask)
        print(f"  sample {idx:02d}: {stats}")

        ax = axes[idx // n_cols, idx % n_cols]
        display = _overlay(image, mask)
        ax.imshow(display)
        ax.set_title(f"sample {idx:02d}")
        ax.axis("off")

    for idx in range(args.n_samples, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved augmented batch grid to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
