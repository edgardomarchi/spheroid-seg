"""End-to-end tests for the evaluation CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from spheroid_seg.data.synthetic import write_synthetic_pair


def _write_config(tmp_path: Path, base_config: Path, overrides: dict | None = None) -> Path:
    """Copy a base config into a temp dir, optionally overriding keys."""
    config = yaml.safe_load(base_config.read_text())
    config["outputs"] = {
        "checkpoints_dir": str(tmp_path / "outputs" / "checkpoints"),
        "logs_dir": str(tmp_path / "outputs" / "logs"),
        "predictions_dir": str(tmp_path / "outputs" / "predictions"),
        "metrics_dir": str(tmp_path / "outputs" / "metrics"),
        "qc_dir": str(tmp_path / "outputs" / "qc"),
        "debug_dir": str(tmp_path / "outputs" / "debug"),
    }
    if overrides:
        config.update(overrides)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _run_train(config_path: Path, epochs: int = 5) -> Path:
    """Train for a few epochs and return the run directory."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spheroid_seg.train",
            "--config",
            str(config_path),
            "--epochs",
            str(epochs),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    config = yaml.safe_load(config_path.read_text())
    runs_dir = Path(config["outputs"]["checkpoints_dir"]).parent / "runs"
    run_dirs = sorted(runs_dir.glob(f"{config_path.stem}_*"), key=lambda p: p.stat().st_mtime)
    assert run_dirs, f"No run directory found under {runs_dir}"
    return run_dirs[-1]


def _run_eval(config_path: Path, split: str = "val", **kwargs) -> subprocess.CompletedProcess:
    """Invoke the eval CLI and return the completed process."""
    cmd = [
        sys.executable,
        "-m",
        "spheroid_seg.eval",
        "--config",
        str(config_path),
        "--split",
        split,
    ]
    for key, value in kwargs.items():
        cmd.append(f"--{key.replace('_', '-')}")
        cmd.append(str(value))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def trained_smoke_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Train a tiny model once per module and return (config_path, run_dir)."""
    tmp_path = tmp_path_factory.mktemp("smoke_run")
    config_path = _write_config(tmp_path, Path("configs/tiny.yaml"))
    run_dir = _run_train(config_path, epochs=2)
    return config_path, run_dir


def test_eval_cli_creates_all_outputs(trained_smoke_run: tuple[Path, Path]) -> None:
    """Eval produces metrics.json, metrics.csv, confusion_matrix.csv, and overlay grid."""
    config_path, _ = trained_smoke_run
    result = _run_eval(config_path, split="val")
    assert result.returncode == 0, result.stderr

    config = yaml.safe_load(config_path.read_text())
    evals_dir = Path(config["outputs"]["checkpoints_dir"]).parent / "evals"
    eval_dirs = sorted(evals_dir.glob(f"{config_path.stem}_*"), key=lambda p: p.stat().st_mtime)
    assert eval_dirs, "No eval output directory created"
    eval_dir = eval_dirs[-1]

    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "metrics.csv").exists()
    assert (eval_dir / "confusion_matrix.csv").exists()
    assert (eval_dir / "overlays_grid.png").exists()


def test_eval_cli_metrics_json_structure(trained_smoke_run: tuple[Path, Path]) -> None:
    """metrics.json contains the required nested report sections."""
    config_path, _ = trained_smoke_run
    result = _run_eval(config_path, split="val")
    assert result.returncode == 0, result.stderr

    config = yaml.safe_load(config_path.read_text())
    evals_dir = Path(config["outputs"]["checkpoints_dir"]).parent / "evals"
    eval_dir = sorted(evals_dir.glob(f"{config_path.stem}_*"), key=lambda p: p.stat().st_mtime)[-1]
    metrics = json.loads((eval_dir / "metrics.json").read_text())

    assert "overall" in metrics
    assert "per_magnification" in metrics
    assert "per_image" in metrics
    assert "confusion_matrix" in metrics
    assert "class_names" in metrics

    overall = metrics["overall"]
    for key in ("dice", "iou"):
        assert key in overall
        assert len(overall[key]) == config["num_classes"]
        assert all(0.0 <= v <= 1.0 for v in overall[key])

    assert "4x" in metrics["per_magnification"]
    assert "10x" in metrics["per_magnification"]


def test_eval_cli_deterministic(trained_smoke_run: tuple[Path, Path]) -> None:
    """Two identical eval invocations produce identical metrics.json and overlay."""
    config_path, _ = trained_smoke_run

    result1 = _run_eval(config_path, split="val")
    result2 = _run_eval(config_path, split="val")
    assert result1.returncode == 0, result1.stderr
    assert result2.returncode == 0, result2.stderr

    config = yaml.safe_load(config_path.read_text())
    evals_dir = Path(config["outputs"]["checkpoints_dir"]).parent / "evals"
    eval_dirs = sorted(evals_dir.glob(f"{config_path.stem}_*"), key=lambda p: p.stat().st_mtime)
    assert len(eval_dirs) >= 2
    dir1, dir2 = eval_dirs[-2], eval_dirs[-1]

    assert (dir1 / "metrics.json").read_bytes() == (dir2 / "metrics.json").read_bytes()

    img1 = cv2.imread(str(dir1 / "overlays_grid.png"))
    img2 = cv2.imread(str(dir2 / "overlays_grid.png"))
    np.testing.assert_array_equal(img1, img2)


def test_eval_cli_invalid_split_exits_nonzero(trained_smoke_run: tuple[Path, Path]) -> None:
    """An invalid --split value is rejected with a non-zero exit."""
    config_path, _ = trained_smoke_run
    result = _run_eval(config_path, split="foo")
    assert result.returncode != 0
    assert "split" in result.stderr.lower()


def test_eval_cli_unresolvable_checkpoint_exits_nonzero(tmp_path: Path) -> None:
    """Eval exits non-zero when no checkpoint can be resolved."""
    config_path = _write_config(tmp_path, Path("configs/tiny.yaml"))
    result = _run_eval(config_path, split="val", checkpoint="/does/not/exist.msgpack")
    assert result.returncode != 0
    assert "checkpoint" in result.stderr.lower()


def test_eval_cli_config_checkpoint_mismatch(
    trained_smoke_run: tuple[Path, Path], tmp_path: Path
) -> None:
    """Loading a checkpoint into an incompatible model exits non-zero."""
    config_path, run_dir = trained_smoke_run
    checkpoint_path = run_dir / "checkpoints" / "best_checkpoint.msgpack"

    bad_config_path = _write_config(
        tmp_path, Path("configs/tiny.yaml"), overrides={"num_classes": 2}
    )
    result = _run_eval(bad_config_path, split="val", checkpoint=str(checkpoint_path))
    assert result.returncode != 0
    assert "incompatible" in result.stderr.lower() or "shape" in result.stderr.lower()


def test_synthetic_fallback_with_committed_splits_no_real_data(tmp_path: Path) -> None:
    """Committed split files are ignored when no real raw/mask pairs exist.

    Regression test for the bug where ``data/splits/*.txt`` are committed and
    reference real image names, but on a clean checkout (CI, fresh clone) no
    real raw/mask pairs are present. In that case the pipeline must fall back
    to the synthetic dataset and its internal split, ignoring the split files.
    """
    data_dir = tmp_path / "data"
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True)

    # Duplicate names across train/val would trigger the leak detector if the
    # split files were read, so training success proves they were ignored.
    (splits_dir / "train.txt").write_text("committed_missing_4x\n")
    (splits_dir / "val.txt").write_text("committed_missing_4x\n")
    (splits_dir / "test.txt").write_text("committed_missing_10x\n")

    # raw/mask dirs are intentionally absent.
    config_path = _write_config(
        tmp_path,
        Path("configs/tiny.yaml"),
        overrides={
            "data": {
                "raw_dir": str(data_dir / "raw"),
                "masks_dir": str(data_dir / "masks"),
                "splits_dir": str(splits_dir),
                "slimia_dir": str(data_dir / "slimia"),
            },
            "synthetic_n_images": 8,
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spheroid_seg.train",
            "--config",
            str(config_path),
            "--epochs",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    combined_output = result.stdout + result.stderr
    assert "synthetic" in combined_output.lower()
    assert "ignored" in combined_output.lower() or "no real" in combined_output.lower()

    config = yaml.safe_load(config_path.read_text())
    runs_dir = Path(config["outputs"]["checkpoints_dir"]).parent / "runs"
    run_dirs = sorted(runs_dir.glob(f"{config_path.stem}_*"), key=lambda p: p.stat().st_mtime)
    assert run_dirs, "No training run directory was created"
    run_dir = run_dirs[-1]

    eval_result = _run_eval(config_path, split="val", run_dir=str(run_dir))
    assert eval_result.returncode == 0, eval_result.stderr


def test_real_data_without_splits_raises_clear_error(tmp_path: Path) -> None:
    """Real raw/mask pairs without split files produce a clear error."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    masks_dir = data_dir / "masks"
    raw_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    # Create a valid raw/mask pair but omit split files.
    write_synthetic_pair(raw_dir, masks_dir, "real_4x", shape=(128, 128))

    config_path = _write_config(
        tmp_path,
        Path("configs/tiny.yaml"),
        overrides={
            "data": {
                "raw_dir": str(raw_dir),
                "masks_dir": str(masks_dir),
                "splits_dir": str(data_dir / "splits"),
                "slimia_dir": str(data_dir / "slimia"),
            },
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spheroid_seg.train",
            "--config",
            str(config_path),
            "--epochs",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "split files" in combined_output.lower()
