"""Tests for pooled validation Dice in the training loop.

Regression coverage for the validation-Dice inflation bug: per-batch macro
Dice gives a free 1.0 to classes absent from a batch, which inflated the
minority-class scores that early stopping used to select the best model.
"""

from __future__ import annotations

import csv
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml

import spheroid_seg.train as train_module
from spheroid_seg.eval import accumulate_confusion_matrix, class_metrics_from_confusion
from spheroid_seg.metrics import dice_score
from spheroid_seg.train import TrainState, create_train_state, evaluate, load_config, train

NUM_CLASSES = 3

NEW_FIELDNAMES = [
    "epoch",
    "train_loss",
    "val_loss",
    "dice_class_0",
    "dice_class_1",
    "dice_class_2",
    "mean_dice",
    "val_dice_pooled_background",
    "val_dice_pooled_loose_cell",
    "val_dice_pooled_aggregate",
    "val_dice_pooled_mean",
]


def _all_background_state() -> TrainState:
    """A TrainState whose apply_fn predicts background everywhere.

    Lets us drive ``evaluate`` with fully controlled ground-truth masks: the
    per-batch predictions are known (all class 0) without training a model.
    """

    def apply_fn(variables, images, train, mutable):
        logits = jnp.full(images.shape[:3] + (NUM_CLASSES,), -10.0)
        return logits.at[..., 0].set(10.0)

    return TrainState.create(
        apply_fn=apply_fn,
        params={"dummy": jnp.zeros(())},
        batch_stats={},
        tx=optax.adamw(1e-3),
    )


def _minority_class_val_arrays(
    n_patches: int = 16,
    size: int = 16,
    minority_patch: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validation arrays where the loose class appears in only one batch.

    Every patch holds an aggregate square; only ``minority_patch`` additionally
    holds a loose-cell square. With all-background predictions, per-batch macro
    Dice scores the loose class a free 1.0 in every batch but one, while the
    pooled loose Dice is ~0.
    """
    if minority_patch is None:
        minority_patch = n_patches - 2
    masks = np.zeros((n_patches, size, size), dtype=np.int32)
    masks[:, 0:4, 0:4] = 2  # aggregate present in every patch
    masks[minority_patch, 8:12, 8:12] = 1  # loose cells in a single patch
    images = np.zeros((n_patches, size, size, 1), dtype=np.float32)
    return images, masks


def test_pooled_dice_matches_eval_path_counts_and_macro_is_inflated() -> None:
    """Pooled validation Dice equals the eval-path counts; macro is inflated.

    Regression test for the inflation bug: the minority class is absent from
    most batches (GT and prediction) and has poor overlap where present, so
    the macro-averaged score is much higher than the pooled score. The pooled
    score must match per-class Dice recomputed from the uint32 confusion
    accumulated over the whole array (``accumulate_confusion_matrix`` /
    ``class_metrics_from_confusion`` semantics, as used by the eval path).
    """
    batch_size = 2
    images, masks = _minority_class_val_arrays()
    state = _all_background_state()
    class_weights = jnp.array([0.5, 1.0, 1.0], dtype=jnp.float32)

    _loss, macro_dice, pooled_dice = evaluate(
        state,
        images,
        masks,
        batch_size,
        class_weights,
        NUM_CLASSES,
        np.random.default_rng(0),
    )

    # Pooled per-class Dice matches the eval path over the whole array.
    all_background = np.zeros_like(masks)
    expected_conf = accumulate_confusion_matrix(all_background, masks, NUM_CLASSES)
    expected_pooled = class_metrics_from_confusion(expected_conf)["dice"]
    np.testing.assert_allclose(np.asarray(pooled_dice), np.asarray(expected_pooled), atol=1e-6)

    # The pooled loose-cell Dice is ~0 (16 target pixels, none predicted).
    assert float(pooled_dice[1]) == pytest.approx(0.0, abs=1e-6)
    # ... while the macro-averaged loose Dice gets a free 1.0 in 7 of 8 batches.
    assert float(macro_dice[1]) == pytest.approx(7.0 / 8.0, abs=1e-6)
    # Aggregate Dice is ~0 under both conventions (present everywhere, missed).
    assert float(pooled_dice[2]) == pytest.approx(0.0, abs=1e-6)
    assert float(macro_dice[2]) == pytest.approx(0.0, abs=1e-6)
    # Macro mean is visibly inflated relative to the pooled mean.
    assert float(jnp.mean(macro_dice)) > float(jnp.mean(pooled_dice))

    # The kept diagnostic is exactly the mean over per-batch dice_score values.
    per_batch = [
        dice_score(
            jnp.zeros((batch_size, 16, 16), dtype=jnp.int32),
            jnp.asarray(masks[start : start + batch_size]),
            NUM_CLASSES,
        )
        for start in range(0, len(masks), batch_size)
    ]
    expected_macro = jnp.mean(jnp.stack(per_batch), axis=0)
    np.testing.assert_allclose(np.asarray(macro_dice), np.asarray(expected_macro), atol=1e-6)


def test_pooled_dice_single_batch_matches_dice_score() -> None:
    """With a single batch, pooled Dice equals dice_score on that batch."""
    images, masks = _minority_class_val_arrays(n_patches=2)
    state = _all_background_state()
    class_weights = jnp.array([0.5, 1.0, 1.0], dtype=jnp.float32)

    _loss, _macro, pooled_dice = evaluate(
        state,
        images,
        masks,
        batch_size=2,
        class_weights=class_weights,
        num_classes=NUM_CLASSES,
        rng=np.random.default_rng(0),
    )

    direct = dice_score(jnp.zeros((2, 16, 16), dtype=jnp.int32), jnp.asarray(masks), NUM_CLASSES)
    np.testing.assert_allclose(np.asarray(pooled_dice), np.asarray(direct), atol=1e-6)


def test_pooled_dice_empty_class_convention() -> None:
    """A class absent from both pooled prediction and target scores 1.0."""
    # Masks contain only background and loose cells: aggregate is absent from
    # both GT and (all-background) prediction, so its pooled Dice must be 1.0.
    masks = np.zeros((4, 16, 16), dtype=np.int32)
    masks[:, 0:4, 0:4] = 1
    images = np.zeros((4, 16, 16, 1), dtype=np.float32)
    state = _all_background_state()
    class_weights = jnp.array([0.5, 1.0, 1.0], dtype=jnp.float32)

    _loss, _macro, pooled_dice = evaluate(
        state,
        images,
        masks,
        2,
        class_weights,
        NUM_CLASSES,
        np.random.default_rng(0),
    )

    assert float(pooled_dice[2]) == pytest.approx(1.0, abs=1e-6)
    # Loose cells are present in the target but never predicted -> 0.0.
    assert float(pooled_dice[1]) == pytest.approx(0.0, abs=1e-6)


def _write_legacy_log(log_path: Path, rows: list[dict]) -> None:
    """Write a train_log.csv in the pre-pooled-Dice (legacy) column layout."""
    legacy_fieldnames = NEW_FIELDNAMES[: 4 + NUM_CLASSES]
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=legacy_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _legacy_row(epoch: int) -> dict:
    return {
        "epoch": epoch,
        "train_loss": f"{1.0 / epoch:.6f}",
        "val_loss": f"{2.0 / epoch:.6f}",
        "dice_class_0": "0.990000",
        "dice_class_1": "0.433000",
        "dice_class_2": "0.600000",
        "mean_dice": "0.674333",
    }


def test_migrate_log_header_pads_legacy_rows(tmp_path: Path) -> None:
    """A legacy CSV gains the pooled columns; old values stay in place."""
    log_path = tmp_path / "train_log.csv"
    _write_legacy_log(log_path, [_legacy_row(1), _legacy_row(2)])

    train_module._migrate_log_header(log_path, NEW_FIELDNAMES)

    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == NEW_FIELDNAMES
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["epoch"] == "1"
    assert rows[0]["mean_dice"] == "0.674333"
    # Epochs logged before the change have no pooled values.
    for column in NEW_FIELDNAMES[7:]:
        assert rows[0][column] == ""
        assert rows[1][column] == ""


def test_migrate_log_header_handles_header_only_file(tmp_path: Path) -> None:
    """A legacy CSV with no data rows migrates to the new header."""
    log_path = tmp_path / "train_log.csv"
    _write_legacy_log(log_path, [])

    train_module._migrate_log_header(log_path, NEW_FIELDNAMES)

    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == NEW_FIELDNAMES
        assert list(reader) == []


def test_migrate_log_header_noop_on_current_header(tmp_path: Path) -> None:
    """A CSV already in the new layout is left untouched."""
    log_path = tmp_path / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NEW_FIELDNAMES)
        writer.writeheader()
        writer.writerow(dict.fromkeys(NEW_FIELDNAMES, "1"))
    before = log_path.read_text()

    train_module._migrate_log_header(log_path, NEW_FIELDNAMES)

    assert log_path.read_text() == before


def test_migrate_log_header_refuses_unknown_header(tmp_path: Path) -> None:
    """An unrecognized header layout is rejected with a clear error."""
    log_path = tmp_path / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mystery_column"])
        writer.writerow([1, 2])

    with pytest.raises(ValueError, match="header"):
        train_module._migrate_log_header(log_path, NEW_FIELDNAMES)


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
    config["patches_per_image"] = 4
    config["early_stopping_patience"] = 2
    config["epochs"] = 3
    return config


def test_model_selection_uses_pooled_mean_dice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Early stopping and best-checkpoint selection follow pooled, not macro."""
    config = _tiny_config(tmp_path)

    # Macro keeps improving while pooled degrades; only pooled must matter.
    scripted = [
        (0.5, jnp.array([0.90, 0.80, 0.70]), jnp.array([0.70, 0.60, 0.50])),  # pooled mean 0.60
        (0.4, jnp.array([0.95, 0.90, 0.85]), jnp.array([0.50, 0.40, 0.30])),  # pooled mean 0.40
        (0.3, jnp.array([0.99, 0.95, 0.90]), jnp.array([0.40, 0.30, 0.20])),  # pooled mean 0.30
    ]
    calls = iter(scripted)

    def fake_evaluate(*args, **kwargs):
        return next(calls)

    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)

    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir)

    log_path = run_dir / "logs" / "train_log.csv"
    with log_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    # All epochs ran; the CSV carries both metrics, existing columns unchanged.
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3]
    assert [row["mean_dice"] for row in rows] == ["0.800000", "0.900000", "0.946667"]
    assert [row["val_dice_pooled_mean"] for row in rows] == ["0.600000", "0.400000", "0.300000"]

    # The stdout line shows macro and pooled side by side.
    out = capsys.readouterr().out
    assert "mean_dice(macro)=0.8000" in out
    assert "pooled_mean_dice=0.6000" in out

    # Selection state follows the pooled mean: best was epoch 1 with 0.60.
    meta_path = run_dir / "checkpoints" / "training_state_metadata.yaml"
    metadata = yaml.safe_load(meta_path.read_text())
    assert metadata["best_dice"] == pytest.approx(0.60)
    assert metadata["selection_metric"] == "val_dice_pooled_mean"

    state = create_train_state(config, jax.random.PRNGKey(0), steps_per_epoch=1)
    with (run_dir / "checkpoints" / "best_checkpoint.msgpack").open("rb") as f:
        best = flax.serialization.from_bytes(
            {"params": state.params, "batch_stats": state.batch_stats, "epoch": 0}, f.read()
        )
    assert best["epoch"] == 1


def _simulate_legacy_run(run_dir: Path) -> None:
    """Turn a current-format run directory into a pre-change (legacy) one.

    Rewrites the CSV in the legacy column layout and strips the
    ``selection_metric`` marker from the checkpoint metadata, exactly as an
    on-disk run created before this change would look.
    """
    log_path = run_dir / "logs" / "train_log.csv"
    with log_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    _write_legacy_log(log_path, rows)

    meta_path = run_dir / "checkpoints" / "training_state_metadata.yaml"
    metadata = yaml.safe_load(meta_path.read_text())
    assert "selection_metric" in metadata
    del metadata["selection_metric"]
    meta_path.write_text(yaml.safe_dump(metadata))


def test_resume_migrates_legacy_csv_and_recomputes_best_dice(tmp_path: Path) -> None:
    """Resuming a pre-change run migrates its CSV and re-derives best_dice."""
    config = _tiny_config(tmp_path)
    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir, epochs_override=2)
    _simulate_legacy_run(run_dir)

    ckpt_path = run_dir / "checkpoints" / "training_state.msgpack"
    train(dict(config), run_dir=run_dir, resume_from=ckpt_path, epochs_override=3)

    # The header was migrated: old rows keep their values, pooled fields empty.
    log_path = run_dir / "logs" / "train_log.csv"
    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == NEW_FIELDNAMES
        rows = list(reader)
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3]
    assert rows[0]["mean_dice"] != ""
    for column in NEW_FIELDNAMES[7:]:
        assert rows[0][column] == ""
        assert rows[1][column] == ""
        assert rows[2][column] != ""

    # The legacy (macro-based) best_dice was replaced by the pooled Dice of
    # the legacy best checkpoint evaluated on the saved validation patches.
    metadata = yaml.safe_load(
        (run_dir / "checkpoints" / "training_state_metadata.yaml").read_text()
    )
    assert metadata["selection_metric"] == "val_dice_pooled_mean"
    assert 0.0 <= metadata["best_dice"] <= 1.0


def test_resume_rejects_unknown_selection_metric(tmp_path: Path) -> None:
    """A checkpoint with an unrecognized selection metric fails loudly."""
    config = _tiny_config(tmp_path)
    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir, epochs_override=2)

    meta_path = run_dir / "checkpoints" / "training_state_metadata.yaml"
    metadata = yaml.safe_load(meta_path.read_text())
    metadata["selection_metric"] = "mystery_metric"
    meta_path.write_text(yaml.safe_dump(metadata))

    ckpt_path = run_dir / "checkpoints" / "training_state.msgpack"
    with pytest.raises(ValueError, match="selection metric"):
        train(dict(config), run_dir=run_dir, resume_from=ckpt_path, epochs_override=3)
