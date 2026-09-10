"""Tests for the training loop."""

from __future__ import annotations

import csv
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from spheroid_seg.train import (
    TrainState,
    create_train_state,
    load_checkpoint,
    load_config,
    train,
    train_step,
)


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


def _tiny_config(tmp_path: Path) -> dict:
    """Return a tiny CPU config that forces the synthetic fallback."""
    config = load_config(Path("configs/tiny.yaml"))
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
    config["epochs"] = 6
    return config


def _read_log(run_dir: Path) -> list[dict]:
    """Read the training log CSV."""
    log_path = run_dir / "logs" / "train_log.csv"
    return list(csv.DictReader(log_path.open("r", newline="")))


def test_resume_equivalence(tmp_path: Path) -> None:
    """A resumed run is bit-identical to an uninterrupted run."""
    config = _tiny_config(tmp_path)
    total_epochs = 6
    resume_after = 3

    run_uninterrupted = tmp_path / "run_uninterrupted"
    train(dict(config), run_dir=run_uninterrupted, epochs_override=total_epochs)

    run_resumed = tmp_path / "run_resumed"
    train(dict(config), run_dir=run_resumed, epochs_override=resume_after)
    ckpt_path = run_resumed / "checkpoints" / "training_state.msgpack"
    assert ckpt_path.exists()
    train(
        dict(config),
        run_dir=run_resumed,
        resume_from=ckpt_path,
        epochs_override=total_epochs,
    )

    log_uninterrupted = _read_log(run_uninterrupted)
    log_resumed = _read_log(run_resumed)
    assert len(log_uninterrupted) == len(log_resumed) == total_epochs
    for row_uninterrupted, row_resumed in zip(log_uninterrupted, log_resumed, strict=True):
        assert row_uninterrupted == row_resumed


def test_training_state_payload_complete(tmp_path: Path) -> None:
    """The resume checkpoint contains every required state component."""
    config = _tiny_config(tmp_path)
    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir, epochs_override=2)

    meta_path = run_dir / "checkpoints" / "training_state_metadata.yaml"
    assert meta_path.exists()
    metadata = yaml.safe_load(meta_path.read_text())
    required_meta_keys = [
        "epoch",
        "best_dice",
        "patience_counter",
        "np_rng_state",
        "jax_rng_key",
        "config",
    ]
    for key in required_meta_keys:
        assert key in metadata, f"Missing metadata key: {key}"

    ckpt_path = run_dir / "checkpoints" / "training_state.msgpack"
    state, loaded_metadata, _patches = load_checkpoint(ckpt_path, dict(config))
    assert isinstance(state, TrainState)
    assert loaded_metadata["epoch"] == metadata["epoch"]


def test_resume_log_continuity(tmp_path: Path) -> None:
    """Resuming appends to the existing CSV without duplicating the header."""
    config = _tiny_config(tmp_path)
    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir, epochs_override=3)

    ckpt_path = run_dir / "checkpoints" / "training_state.msgpack"
    assert ckpt_path.exists()
    train(dict(config), run_dir=run_dir, resume_from=ckpt_path, epochs_override=6)

    log_path = run_dir / "logs" / "train_log.csv"
    text = log_path.read_text()
    header_rows = [line for line in text.splitlines() if line.startswith("epoch,")]
    assert len(header_rows) == 1

    rows = _read_log(run_dir)
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3, 4, 5, 6]


def test_resume_does_not_create_new_run_directory(tmp_path: Path) -> None:
    """Resuming continues in the same run directory."""
    config = _tiny_config(tmp_path)
    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir, epochs_override=2)

    ckpt_path = run_dir / "checkpoints" / "training_state.msgpack"
    initial_mtime = run_dir.stat().st_mtime
    train(dict(config), run_dir=run_dir, resume_from=ckpt_path, epochs_override=4)

    # No additional run directories should appear under tmp_path.
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("run")]
    assert run_dirs == [run_dir]
    assert run_dir.stat().st_mtime >= initial_mtime


def test_resume_missing_checkpoint(tmp_path: Path) -> None:
    """--resume with a nonexistent path fails with a clear error."""
    config = _tiny_config(tmp_path)
    missing = tmp_path / "does_not_exist.msgpack"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        train(dict(config), run_dir=tmp_path / "run", resume_from=missing)


def test_resume_old_format_checkpoint(tmp_path: Path) -> None:
    """--resume with a legacy shallow checkpoint fails with a clear error."""
    config = _tiny_config(tmp_path)
    state = create_train_state(config, jax.random.PRNGKey(config["seed"]), steps_per_epoch=1)
    legacy = {
        "params": state.params,
        "batch_stats": state.batch_stats,
        "epoch": 2,
    }
    ckpt_path = tmp_path / "legacy.msgpack"
    with ckpt_path.open("wb") as f:
        f.write(flax.serialization.to_bytes(legacy))

    with pytest.raises(ValueError, match="Old-format checkpoint"):
        train(dict(config), run_dir=tmp_path / "run", resume_from=ckpt_path)


def test_resume_mismatched_config(tmp_path: Path) -> None:
    """--resume rejects immutable config changes with a clear error."""
    config = _tiny_config(tmp_path)
    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir, epochs_override=2)

    ckpt_path = run_dir / "checkpoints" / "training_state.msgpack"
    bad_config = dict(config)
    bad_config["base_features"] = 999
    with pytest.raises(ValueError, match="base_features"):
        train(bad_config, run_dir=tmp_path / "run2", resume_from=ckpt_path)
