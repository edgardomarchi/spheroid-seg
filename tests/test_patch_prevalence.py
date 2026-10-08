"""Tests for exact patch class-prevalence counting and its training integration."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spheroid_seg.patch_prevalence import (
    CSV_COLUMNS,
    PREVALENCE_CSV_NAME,
    patch_class_prevalence,
    prevalence_summary,
    write_prevalence_csv,
)
from spheroid_seg.train import load_config, train

NUM_CLASSES = 3

# Four 2x2 patches with known class IDs:
#  - patch 0: a single loose-cell pixel (one-pixel foreground);
#  - patch 1: loose + aggregate pixels (multiple classes per patch);
#  - patch 2: background only (no foreground);
#  - patch 3: aggregate only.
FIXTURE_MASKS = np.array(
    [
        [[0, 0], [0, 1]],
        [[1, 2], [2, 0]],
        [[0, 0], [0, 0]],
        [[2, 2], [2, 2]],
    ],
    dtype=np.int32,
)

# Second fixture with different composition so train/val rows are independent.
VAL_MASKS = np.array(
    [
        [[0, 1], [1, 1]],
        [[0, 0], [0, 2]],
    ],
    dtype=np.int32,
)


def _read_prevalence_csv(path: Path) -> list[dict[str, str]]:
    """Read the prevalence CSV and return its rows."""
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _rows_by_split_class(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    """Index prevalence rows by (split, class_name)."""
    return {(row["split"], row["class_name"]): row for row in rows}


def _expected_stats(masks: np.ndarray) -> dict[str, Any]:
    """Independent reference implementation of the prevalence statistics."""
    n_patches = len(masks)
    total_pixels = int(masks.size)
    pixel_counts = np.bincount(masks.reshape(-1), minlength=NUM_CLASSES)
    patches_with_class = np.zeros(NUM_CLASSES, dtype=np.int64)
    object_patch_flags: list[np.ndarray] = []
    for patch in masks:
        patch_counts = np.bincount(patch.reshape(-1), minlength=NUM_CLASSES)
        patches_with_class += patch_counts > 0
        object_patch_flags.append(patch_counts[1:] > 0)
    return {
        "pixel_counts": pixel_counts,
        "patches_with_class": patches_with_class,
        "object_pixel_count": int(pixel_counts[1:].sum()),
        "object_patch_count": int(np.count_nonzero(np.any(np.stack(object_patch_flags), axis=1))),
        "n_patches": n_patches,
        "total_pixels": total_pixels,
    }


def _assert_row_matches(
    row: dict[str, str],
    *,
    split: str,
    pixel_count: int,
    patches_with_class_count: int,
    n_patches: int,
    total_pixels: int,
) -> None:
    """Assert one CSV row carries the exact expected values."""
    assert row["split"] == split
    assert int(row["pixel_count"]) == pixel_count
    assert int(row["patches_with_class_count"]) == patches_with_class_count
    assert int(row["n_patches"]) == n_patches
    assert int(row["total_pixels"]) == total_pixels
    expected_pixel_fraction = pixel_count / total_pixels if total_pixels else 0.0
    expected_patch_fraction = patches_with_class_count / n_patches if n_patches else 0.0
    assert float(row["pixel_fraction"]) == pytest.approx(expected_pixel_fraction)
    assert float(row["patches_with_class_fraction"]) == pytest.approx(expected_patch_fraction)


# ---------------------------------------------------------------------------
# Exact counting
# ---------------------------------------------------------------------------


def test_pixel_counts_and_fractions_are_exact(tmp_path: Path) -> None:
    """A synthetic patch set with known class IDs yields exact counts/fractions."""
    stats = patch_class_prevalence(FIXTURE_MASKS, NUM_CLASSES)

    assert stats.pixel_counts.dtype == np.uint64
    assert stats.patches_with_class.dtype == np.uint64
    assert stats.pixel_counts.tolist() == [8, 2, 6]
    assert stats.total_pixels == 16
    assert stats.n_patches == 4

    path = tmp_path / "counts.csv"
    stats_map = write_prevalence_csv(path, {"train": FIXTURE_MASKS}, NUM_CLASSES)
    assert set(stats_map) == {"train"}
    rows = _rows_by_split_class(_read_prevalence_csv(path))

    _assert_row_matches(
        rows[("train", "background")],
        split="train",
        pixel_count=8,
        patches_with_class_count=3,  # patch 3 is aggregate-only
        n_patches=4,
        total_pixels=16,
    )
    _assert_row_matches(
        rows[("train", "loose cell")],
        split="train",
        pixel_count=2,
        patches_with_class_count=2,
        n_patches=4,
        total_pixels=16,
    )
    _assert_row_matches(
        rows[("train", "aggregate")],
        split="train",
        pixel_count=6,
        patches_with_class_count=2,
        n_patches=4,
        total_pixels=16,
    )


def test_object_row_is_exact_union_of_loose_and_aggregate(tmp_path: Path) -> None:
    """The object row counts the exact union, not the sum, of the two classes."""
    stats = patch_class_prevalence(FIXTURE_MASKS, NUM_CLASSES)
    assert stats.object_pixel_count == 2 + 6
    # Patch 1 contains both classes but must be counted once for object.
    assert stats.object_patch_count == 3

    path = tmp_path / "object.csv"
    write_prevalence_csv(path, {"train": FIXTURE_MASKS}, NUM_CLASSES)
    rows = _rows_by_split_class(_read_prevalence_csv(path))

    row = rows[("train", "object")]
    assert row is not None, "object row must be present"
    _assert_row_matches(
        row,
        split="train",
        pixel_count=8,
        patches_with_class_count=3,
        n_patches=4,
        total_pixels=16,
    )


def test_patch_presence_handles_edge_cases() -> None:
    """Presence counts handle one-pixel, multi-class, and no-foreground patches."""
    # A single 1x1 patch with one loose pixel.
    one_pixel = np.array([[[1]]], dtype=np.int32)
    stats = patch_class_prevalence(one_pixel, NUM_CLASSES)
    assert stats.pixel_counts.tolist() == [0, 1, 0]
    assert stats.patches_with_class.tolist() == [0, 1, 0]
    assert stats.object_pixel_count == 1
    assert stats.object_patch_count == 1

    # The no-foreground fixture patch must count only toward background.
    bg_only = FIXTURE_MASKS[2:3]
    stats = patch_class_prevalence(bg_only, NUM_CLASSES)
    assert stats.patches_with_class.tolist() == [1, 0, 0]
    assert stats.object_patch_count == 0
    assert stats.object_pixel_count == 0


def test_empty_class_is_reported_as_zero_not_omitted(tmp_path: Path) -> None:
    """Classes absent from every patch keep a zero-count row."""
    masks = np.array([[[0, 1], [1, 0]]], dtype=np.int32)  # no aggregate pixels
    path = tmp_path / "empty_class.csv"
    write_prevalence_csv(path, {"train": masks}, NUM_CLASSES)
    rows = _rows_by_split_class(_read_prevalence_csv(path))

    row = rows[("train", "aggregate")]
    assert row is not None, "absent classes must keep their row"
    _assert_row_matches(
        row,
        split="train",
        pixel_count=0,
        patches_with_class_count=0,
        n_patches=1,
        total_pixels=4,
    )
    assert float(row["pixel_fraction"]) == 0.0
    assert float(row["patches_with_class_fraction"]) == 0.0


def test_train_and_val_splits_are_reported_independently(tmp_path: Path) -> None:
    """Each split gets its own rows with its own n_patches/total_pixels."""
    path = tmp_path / "splits.csv"
    write_prevalence_csv(path, {"train": FIXTURE_MASKS, "val": VAL_MASKS}, NUM_CLASSES)
    rows = _rows_by_split_class(_read_prevalence_csv(path))

    train_row = rows[("train", "loose cell")]
    val_row = rows[("val", "loose cell")]
    assert int(train_row["pixel_count"]) == 2
    assert int(train_row["n_patches"]) == 4
    assert int(train_row["total_pixels"]) == 16
    assert int(val_row["pixel_count"]) == 3
    assert int(val_row["n_patches"]) == 2
    assert int(val_row["total_pixels"]) == 8


def test_invalid_masks_are_rejected() -> None:
    """Non-integer masks, out-of-range IDs, and non-3D arrays raise ValueError."""
    with pytest.raises(ValueError, match="integer"):
        patch_class_prevalence(FIXTURE_MASKS.astype(np.float32), NUM_CLASSES)
    with pytest.raises(ValueError, match="outside"):
        patch_class_prevalence(np.array([[[0, 3], [0, 0]]], dtype=np.int32), NUM_CLASSES)
    with pytest.raises(ValueError, match=r"N, P, P"):
        patch_class_prevalence(FIXTURE_MASKS[0], NUM_CLASSES)


def test_summary_mentions_foreground_loose_aggregate_and_object_patches() -> None:
    """The stdout summary carries foreground/class fractions and object patch counts."""
    stats = {
        "train": patch_class_prevalence(FIXTURE_MASKS, NUM_CLASSES),
        "val": patch_class_prevalence(VAL_MASKS, NUM_CLASSES),
    }
    text = prevalence_summary(stats, NUM_CLASSES)
    lines = text.splitlines()
    assert lines[0] == "Patch class prevalence:"
    assert "train:" in lines[1]
    assert "val:" in lines[2]
    assert "foreground=0.5000" in lines[1]  # 8/16
    assert "loose=0.1250" in lines[1]  # 2/16
    assert "aggregate=0.3750" in lines[1]  # 6/16
    assert "object patches=3/4" in lines[1]


# ---------------------------------------------------------------------------
# Training integration
# ---------------------------------------------------------------------------


def _tiny_synthetic_config(tmp_path: Path) -> dict:
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


def _assert_csv_matches_saved_patches(run_dir: Path) -> None:
    """The prevalence CSV must agree exactly with the run's saved patch arrays."""
    prevalence_path = run_dir / "logs" / PREVALENCE_CSV_NAME
    rows = _rows_by_split_class(_read_prevalence_csv(prevalence_path))
    with np.load(run_dir / "checkpoints" / "training_patches.npz") as patches:
        for split, masks_key in (("train", "train_masks"), ("val", "val_masks")):
            expected = _expected_stats(np.asarray(patches[masks_key]))
            for class_index, class_name in enumerate(["background", "loose cell", "aggregate"]):
                _assert_row_matches(
                    rows[(split, class_name)],
                    split=split,
                    pixel_count=int(expected["pixel_counts"][class_index]),
                    patches_with_class_count=int(expected["patches_with_class"][class_index]),
                    n_patches=expected["n_patches"],
                    total_pixels=expected["total_pixels"],
                )
            _assert_row_matches(
                rows[(split, "object")],
                split=split,
                pixel_count=expected["object_pixel_count"],
                patches_with_class_count=expected["object_patch_count"],
                n_patches=expected["n_patches"],
                total_pixels=expected["total_pixels"],
            )


def test_tiny_run_creates_prevalence_csv(tmp_path: Path) -> None:
    """A tiny CPU training run writes logs/patch_class_prevalence.csv."""
    config = _tiny_synthetic_config(tmp_path)
    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir, epochs_override=2)

    prevalence_path = run_dir / "logs" / PREVALENCE_CSV_NAME
    assert prevalence_path.is_file(), "prevalence CSV must be written for fresh runs"

    with prevalence_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == CSV_COLUMNS

    _assert_csv_matches_saved_patches(run_dir)


def test_prevalence_csv_counts_match_saved_patches_exactly(tmp_path: Path) -> None:
    """The CSV describes the exact augmented patches saved in the checkpoint."""
    config = _tiny_synthetic_config(tmp_path)
    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir, epochs_override=3)
    _assert_csv_matches_saved_patches(run_dir)


def test_deterministic_runs_produce_identical_losses_and_prevalence(tmp_path: Path) -> None:
    """Two identical smoke runs match in losses and in the prevalence CSV."""
    config = _tiny_synthetic_config(tmp_path)

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    train(dict(config), run_dir=run_a, epochs_override=3)
    train(dict(config), run_dir=run_b, epochs_override=3)

    log_a = (run_a / "logs" / "train_log.csv").read_text()
    log_b = (run_b / "logs" / "train_log.csv").read_text()
    assert log_a == log_b, "losses must be bit-identical across identical runs"

    prev_a = (run_a / "logs" / PREVALENCE_CSV_NAME).read_text()
    prev_b = (run_b / "logs" / PREVALENCE_CSV_NAME).read_text()
    assert prev_a == prev_b, "prevalence CSV must be identical across identical runs"


def test_resume_preserves_prevalence_csv(tmp_path: Path) -> None:
    """Resuming a run leaves the prevalence CSV untouched (no duplicate/corrupt rows)."""
    config = _tiny_synthetic_config(tmp_path)
    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir, epochs_override=3)

    prevalence_path = run_dir / "logs" / PREVALENCE_CSV_NAME
    content_before = prevalence_path.read_text()
    _assert_csv_matches_saved_patches(run_dir)

    ckpt_path = run_dir / "checkpoints" / "training_state.msgpack"
    train(dict(config), run_dir=run_dir, resume_from=ckpt_path, epochs_override=6)

    assert prevalence_path.read_text() == content_before, (
        "resume must not rewrite the prevalence CSV"
    )
    _assert_csv_matches_saved_patches(run_dir)

    # Sanity: the resumed run actually continued (log continuity is intact).
    with (run_dir / "logs" / "train_log.csv").open("r", newline="") as f:
        epochs = [int(row["epoch"]) for row in csv.DictReader(f)]
    assert epochs == [1, 2, 3, 4, 5, 6]


def test_prevalence_summary_printed_once_for_fresh_and_never_on_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Fresh runs print the summary once; resume runs print it never."""
    config = _tiny_synthetic_config(tmp_path)
    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir, epochs_override=2)
    out = capsys.readouterr().out
    assert out.count("Patch class prevalence:") == 1
    assert "train:" in out and "val:" in out
    assert "object patches=" in out

    capsys.readouterr()
    ckpt_path = run_dir / "checkpoints" / "training_state.msgpack"
    train(dict(config), run_dir=run_dir, resume_from=ckpt_path, epochs_override=4)
    out = capsys.readouterr().out
    assert "Patch class prevalence:" not in out


def test_checkpoint_schema_unchanged_by_prevalence_logging(tmp_path: Path) -> None:
    """Prevalence logging adds no checkpoint artifacts; the schema stays fixed."""
    config = _tiny_synthetic_config(tmp_path)
    run_dir = tmp_path / "run"
    train(dict(config), run_dir=run_dir, epochs_override=2)

    checkpoint_files = {p.name for p in (run_dir / "checkpoints").iterdir()}
    assert checkpoint_files == {
        "training_state.msgpack",
        "training_state_metadata.yaml",
        "training_patches.npz",
        "best_checkpoint.msgpack",
    }


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------


def test_colab_bgweight05_config_loads_with_background_weight_05() -> None:
    """The new experiment config loads and sets class_weights to [0.5, 1.0, 1.0]."""
    config_path = Path("configs/colab_bgweight05.yaml")
    assert config_path.is_file(), "configs/colab_bgweight05.yaml must exist"
    config = load_config(config_path)
    assert config["class_weights"] == [0.5, 1.0, 1.0]
    # The experiment must not touch the object-patch sampling ratio.
    assert config["object_patch_ratio"] == 0.8


def test_colab_bgweight05_matches_colab_except_class_weights() -> None:
    """The experiment config preserves every effective value of configs/colab.yaml."""
    colab = load_config(Path("configs/colab.yaml"))
    experiment = load_config(Path("configs/colab_bgweight05.yaml"))

    colab_weights = colab.pop("class_weights")
    experiment_weights = experiment.pop("class_weights")
    assert experiment == colab, (
        "configs/colab_bgweight05.yaml must differ from configs/colab.yaml only in class_weights"
    )
    assert colab_weights == [0.1, 1.0, 1.0]
    assert experiment_weights == [0.5, 1.0, 1.0]
