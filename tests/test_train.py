"""Tests for the training loop."""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spheroid_seg.train import create_train_state, load_config, train, train_step


def _fixed_batch(
    rng: np.random.Generator,
    batch_size: int = 2,
    size: int = 128,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create a fixed synthetic batch with a few labeled circles."""
    images = rng.random((batch_size, size, size, 1)).astype(np.float32)
    masks = np.zeros((batch_size, size, size), dtype=np.int32)
    for b in range(batch_size):
        masks[b, size // 4 : size // 2, size // 4 : size // 2] = 1
        masks[b, size // 2 : 3 * size // 4, size // 2 : 3 * size // 4] = 2
    return jnp.array(images), jnp.array(masks)


def test_train_step_reduces_loss() -> None:
    """A single training step reduces the loss on a fixed batch."""
    rng = np.random.default_rng(42)
    config = {
        "num_classes": 3,
        "base_features": 8,
        "input_channels": "grayscale",
        "patch_size": 128,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "epochs": 1,
        "batch_size": 2,
        "class_weights": [0.1, 1.0, 1.0],
        "seed": 0,
    }

    batch = _fixed_batch(rng, batch_size=config["batch_size"], size=config["patch_size"])
    class_weights = jnp.array(config["class_weights"], dtype=jnp.float32)

    state = create_train_state(config, jax.random.PRNGKey(config["seed"]), steps_per_epoch=1)

    losses: list[float] = []
    for _ in range(5):
        state, step_loss = train_step(state, batch, class_weights)
        losses.append(float(step_loss))

    assert losses[-1] < losses[0]


def test_synthetic_training_does_not_diverge(tmp_path: Path) -> None:
    """Synthetic fallback training stays stable: val loss does not explode.

    Regression test for the train/validation divergence bug where BatchNorm
    running statistics lagged the batch statistics used during training,
    causing validation metrics to degrade while training loss decreased.
    """
    config = load_config(Path("configs/tiny.yaml"))

    # Force the synthetic fallback path and use a small batch / down-weighted
    # background to reproduce the eval-mode BatchNorm mismatch on CPU.
    empty_raw = tmp_path / "raw"
    empty_masks = tmp_path / "masks"
    empty_raw.mkdir()
    empty_masks.mkdir()
    config["data"]["raw_dir"] = str(empty_raw)
    config["data"]["masks_dir"] = str(empty_masks)

    config["batch_size"] = 2
    config["class_weights"] = [0.1, 1.0, 1.0]
    config["synthetic_n_images"] = 8
    config["patches_per_image"] = 8
    config["early_stopping_patience"] = 20

    run_dir = tmp_path / "run"
    train(config, run_dir=run_dir, epochs_override=5)

    log_path = run_dir / "logs" / "train_log.csv"
    rows = list(csv.DictReader(log_path.open("r", newline="")))
    assert len(rows) == 5

    initial_train_loss = float(rows[0]["train_loss"])
    final_train_loss = float(rows[-1]["train_loss"])
    initial_val_loss = float(rows[0]["val_loss"])
    final_val_loss = float(rows[-1]["val_loss"])
    final_bg_dice = float(rows[-1]["dice_class_0"])

    assert final_train_loss < initial_train_loss, (
        f"Training loss did not decrease: {initial_train_loss:.4f} -> {final_train_loss:.4f}"
    )
    assert final_val_loss < 2 * initial_val_loss, (
        f"Validation loss diverged: {initial_val_loss:.4f} -> {final_val_loss:.4f}"
    )
    assert final_bg_dice > 0.5, f"Background Dice collapsed: {final_bg_dice:.4f}"
