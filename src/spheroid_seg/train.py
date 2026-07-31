"""Training loop for the spheroid segmentation U-Net."""

from __future__ import annotations

import argparse
import csv
import datetime
import tempfile
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training import train_state

from spheroid_seg.data.augment import apply_augmentation, build_augmentation
from spheroid_seg.data.dataset import SpheroidDataset
from spheroid_seg.data.patching import extract_patches
from spheroid_seg.data.synthetic import generate_synthetic_dataset
from spheroid_seg.losses import segmentation_loss
from spheroid_seg.metrics import dice_score
from spheroid_seg.models.unet import UNet


class TrainState(train_state.TrainState):
    """TrainState that also carries BatchNorm statistics."""

    batch_stats: Any


def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML configuration file."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def _ensure_data_dirs(config: dict[str, Any], tmp_dir: Path) -> tuple[Path, Path]:
    """Return existing raw/mask dirs or generate a synthetic fallback dataset."""
    raw_dir = Path(config["data"]["raw_dir"])
    masks_dir = Path(config["data"]["masks_dir"])

    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    has_raw = raw_dir.exists() and any(
        p.suffix.lower() in image_suffixes for p in raw_dir.iterdir()
    )
    has_masks = masks_dir.exists() and any(
        p.suffix.lower() in image_suffixes for p in masks_dir.iterdir()
    )

    if has_raw and has_masks:
        return raw_dir, masks_dir

    print(
        f"No real data found in '{raw_dir}' / '{masks_dir}'. "
        "Generating synthetic fallback dataset for smoke testing."
    )
    synth_raw = tmp_dir / "raw"
    synth_masks = tmp_dir / "masks"
    return generate_synthetic_dataset(
        synth_raw,
        synth_masks,
        n_images=config.get("synthetic_n_images", 16),
        shape=config.get("synthetic_image_shape", (512, 512)),
        seed=config["seed"],
    )


def _build_patch_arrays(
    dataset: SpheroidDataset,
    config: dict[str, Any],
    rng: np.random.Generator,
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and optionally augment patches from a dataset."""
    patch_size = config["patch_size"]
    min_object_fraction = config["min_object_fraction"]
    object_patch_ratio = config["object_patch_ratio"]
    patches_per_image = config["patches_per_image"]

    transform = build_augmentation(config["augment"], seed=int(rng.integers(0, 2**31)))

    image_patches: list[np.ndarray] = []
    mask_patches: list[np.ndarray] = []

    for idx in range(len(dataset)):
        image, mask, _ = dataset[idx]
        imgs, masks = extract_patches(
            image,
            mask,
            patch_size=patch_size,
            min_object_fraction=min_object_fraction,
            object_patch_ratio=object_patch_ratio,
            patches_per_image=patches_per_image,
            seed=int(rng.integers(0, 2**31)),
        )
        for img_patch, mask_patch in zip(imgs, masks, strict=False):
            if augment:
                img_patch, mask_patch = apply_augmentation(transform, img_patch, mask_patch)
            image_patches.append(img_patch)
            mask_patches.append(mask_patch)

    images = np.stack(image_patches).astype(np.float32)
    masks = np.stack(mask_patches).astype(np.int32)

    # Add channel dimension for grayscale.
    if images.ndim == 3:
        images = images[..., np.newaxis]

    return images, masks


def _batches(
    images: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
    """Yield (image, mask) JAX batches."""
    n = len(images)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        # Drop incomplete final batch.
        if len(idx) < batch_size:
            continue
        yield jnp.array(images[idx]), jnp.array(masks[idx])


def _mean_dice(metrics: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Mean per-class Dice across a list of metric dictionaries."""
    return jnp.mean(metrics["dice"])


def create_train_state(
    config: dict[str, Any],
    rng: jax.Array,
    steps_per_epoch: int,
) -> TrainState:
    """Initialize model parameters, batch stats, and optimizer state."""
    model = UNet(
        num_classes=config["num_classes"],
        base_features=config["base_features"],
        input_channels=config["input_channels"],
    )
    channels = 1 if config["input_channels"] == "grayscale" else 3
    patch_size = config["patch_size"]
    dummy = jnp.ones((1, patch_size, patch_size, channels), dtype=jnp.float32)
    variables = model.init(rng, dummy, train=True)

    epochs = config["epochs"]
    total_steps = max(epochs * steps_per_epoch, 1)
    schedule = optax.cosine_decay_schedule(
        init_value=config["lr"],
        decay_steps=total_steps,
        alpha=0.0,
    )
    tx = optax.adamw(
        learning_rate=schedule,
        weight_decay=config["weight_decay"],
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
    )


@jax.jit
def train_step(
    state: TrainState,
    batch: tuple[jnp.ndarray, jnp.ndarray],
    class_weights: jnp.ndarray,
) -> tuple[TrainState, jnp.ndarray]:
    """Execute one training step and return updated state + loss."""
    images, masks = batch

    def loss_fn(params: dict) -> tuple[jnp.ndarray, dict]:
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            images,
            train=True,
            mutable=["batch_stats"],
        )
        loss = segmentation_loss(logits, masks, class_weights=class_weights)
        return loss, updates

    (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, loss


@partial(jax.jit, static_argnums=(3,))
def eval_step(
    state: TrainState,
    batch: tuple[jnp.ndarray, jnp.ndarray],
    class_weights: jnp.ndarray,
    num_classes: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Execute one evaluation step and return loss + per-class Dice."""
    images, masks = batch
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        images,
        train=False,
        mutable=False,
    )
    loss = segmentation_loss(logits, masks, class_weights=class_weights)
    preds = jnp.argmax(logits, axis=-1)
    dice = dice_score(preds, masks, num_classes=num_classes)
    return loss, dice


def evaluate(
    state: TrainState,
    images: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
    class_weights: jnp.ndarray,
    num_classes: int,
    rng: np.random.Generator,
) -> tuple[float, jnp.ndarray]:
    """Evaluate the model on a patch array and return average loss and Dice."""
    losses: list[float] = []
    dices: list[jnp.ndarray] = []
    for batch in _batches(images, masks, batch_size, rng, shuffle=False):
        loss, dice = eval_step(state, batch, class_weights, num_classes)
        losses.append(float(loss))
        dices.append(dice)

    mean_loss = float(np.mean(losses))
    mean_dice = jnp.mean(jnp.stack(dices), axis=0)
    return mean_loss, mean_dice


def save_checkpoint(
    state: TrainState,
    epoch: int,
    path: Path,
) -> None:
    """Serialize parameters and batch stats to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "params": state.params,
        "batch_stats": state.batch_stats,
        "epoch": epoch,
    }
    with path.open("wb") as f:
        f.write(flax.serialization.to_bytes(ckpt))


def _dataset_from_pairs(
    pairs: list[tuple[Path, Path, str]],
    raw_dir: Path,
    masks_dir: Path,
    input_channels: str,
    class_mapping: dict[int, int],
) -> SpheroidDataset:
    """Create a SpheroidDataset from an explicit pair list."""
    dataset = SpheroidDataset.__new__(SpheroidDataset)
    dataset.pairs = pairs
    dataset.raw_dir = raw_dir
    dataset.masks_dir = masks_dir
    dataset.input_channels = input_channels
    dataset.class_mapping = class_mapping
    return dataset


def train(
    config: dict[str, Any],
    *,
    run_dir: Path,
    overfit_one_batch: bool = False,
    epochs_override: int | None = None,
) -> None:
    """Run the training loop."""
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    seed = config["seed"]
    np_rng = np.random.default_rng(seed)
    jax_rng = jax.random.PRNGKey(seed)

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir, masks_dir = _ensure_data_dirs(config, Path(tmp))

        # Use all synthetic data as train; reserve a validation subset by reusing
        # a fresh synthetic dataset if no real splits exist.
        splits_dir = Path(config["data"]["splits_dir"])
        train_dataset = SpheroidDataset(
            raw_dir,
            masks_dir,
            input_channels=config["input_channels"],
            class_mapping=config["class_mapping"],
        )

        if splits_dir.exists() and any(splits_dir.glob("*.txt")):
            # Future path: load real train/val splits.
            from spheroid_seg.data.splits import load_splits

            splits = load_splits(splits_dir)
            train_names = set(splits["train"])
            val_names = set(splits["val"])
            train_pairs = [p for p in train_dataset.pairs if p[2] in train_names]
            val_pairs = [p for p in train_dataset.pairs if p[2] in val_names]
            train_dataset.pairs = train_pairs
            val_dataset = _dataset_from_pairs(
                val_pairs, raw_dir, masks_dir, config["input_channels"], config["class_mapping"]
            )
        else:
            # Smoke-test path: split the dataset 80/20 by base name.
            all_pairs = train_dataset.pairs
            np_rng.shuffle(all_pairs)
            split = int(0.8 * len(all_pairs))
            train_dataset.pairs = all_pairs[:split]
            val_dataset = _dataset_from_pairs(
                all_pairs[split:],
                raw_dir,
                masks_dir,
                config["input_channels"],
                config["class_mapping"],
            )

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError(
                "Train or validation set is empty. "
                "Provide real data or rely on synthetic fallback."
            )

        print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

        print("Building training patches...")
        train_images, train_masks = _build_patch_arrays(
            train_dataset, config, np_rng, augment=True
        )
        print(f"  {len(train_images)} training patches.")

        print("Building validation patches...")
        val_images, val_masks = _build_patch_arrays(
            val_dataset, config, np_rng, augment=False
        )
        print(f"  {len(val_images)} validation patches.")

        steps_per_epoch = max(len(train_images) // config["batch_size"], 1)
        state = create_train_state(config, jax_rng, steps_per_epoch)

        class_weights = jnp.array(config["class_weights"], dtype=jnp.float32)
        num_classes = config["num_classes"]
        batch_size = config["batch_size"]
        epochs = epochs_override if epochs_override is not None else config["epochs"]

        log_path = logs_dir / "train_log.csv"
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            *{f"dice_class_{c}" for c in range(num_classes)},
            "mean_dice",
        ]
        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        if overfit_one_batch:
            # Take the first full batch and repeatedly optimize on it.
            overfit_batch = next(_batches(train_images, train_masks, batch_size, np_rng))
            initial_loss = float(
                segmentation_loss(
                    state.apply_fn(
                        {"params": state.params, "batch_stats": state.batch_stats},
                        overfit_batch[0],
                        train=False,
                        mutable=False,
                    ),
                    overfit_batch[1],
                    class_weights=class_weights,
                )
            )
            print(f"Overfit-one-batch initial loss: {initial_loss:.6f}")
            for step in range(epochs):
                state, loss = train_step(state, overfit_batch, class_weights)
                if step % max(1, epochs // 10) == 0 or step == epochs - 1:
                    print(f"  step {step:4d}: loss = {float(loss):.6f}")
            final_loss = float(loss)
            print(f"Overfit-one-batch final loss: {final_loss:.6f}")
            save_checkpoint(state, epochs, checkpoints_dir / "overfit_checkpoint.msgpack")
            return

        best_dice = -1.0
        patience_counter = 0
        patience = config["early_stopping_patience"]
        best_ckpt_path = checkpoints_dir / "best_checkpoint.msgpack"

        for epoch in range(1, epochs + 1):
            # Training
            train_losses: list[float] = []
            for batch in _batches(train_images, train_masks, batch_size, np_rng):
                state, loss = train_step(state, batch, class_weights)
                train_losses.append(float(loss))

            # Validation
            val_loss, val_dice = evaluate(
                state,
                val_images,
                val_masks,
                batch_size,
                class_weights,
                num_classes,
                np_rng,
            )
            mean_dice = float(jnp.mean(val_dice))
            train_loss = float(np.mean(train_losses))

            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                **{f"dice_class_{c}": f"{float(val_dice[c]):.6f}" for c in range(num_classes)},
                "mean_dice": f"{mean_dice:.6f}",
            }
            with log_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

            print(
                f"Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} mean_dice={mean_dice:.4f} "
                f"dice={np.array(val_dice)}"
            )

            if mean_dice > best_dice:
                best_dice = mean_dice
                patience_counter = 0
                save_checkpoint(state, epoch, best_ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the spheroid segmentation model.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help="Train on a single fixed synthetic batch for N steps (N = epochs).",
    )
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"{Path(args.config).stem}_{timestamp}"
    run_dir = Path(config["outputs"]["checkpoints_dir"]).parent / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    train(
        config,
        run_dir=run_dir,
        overfit_one_batch=args.overfit_one_batch,
        epochs_override=args.epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
