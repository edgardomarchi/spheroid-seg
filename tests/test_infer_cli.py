"""End-to-end tests for the inference CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from spheroid_seg.data.synthetic import generate_synthetic_dataset


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


def _run_infer(config_path: Path, input_path: Path, **kwargs) -> subprocess.CompletedProcess:
    """Invoke the infer CLI and return the completed process."""
    cmd = [
        sys.executable,
        "-m",
        "spheroid_seg.infer",
        "--config",
        str(config_path),
        "--input",
        str(input_path),
    ]
    for key, value in kwargs.items():
        cmd.append(f"--{key.replace('_', '-')}")
        cmd.append(str(value))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def trained_smoke_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Train a tiny model once per module and return (config_path, run_dir)."""
    tmp_path = tmp_path_factory.mktemp("infer_smoke_run")
    config_path = _write_config(tmp_path, Path("configs/tiny.yaml"))
    run_dir = _run_train(config_path, epochs=2)
    return config_path, run_dir


@pytest.fixture
def synthetic_image_dir(tmp_path: Path) -> Path:
    """Create a directory with one 4x and one 10x synthetic raw image."""
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    generate_synthetic_dataset(raw_dir, masks_dir, n_images=4, shape=(256, 256), seed=42)
    return raw_dir


def test_infer_cli_creates_mask_and_overlay_for_file(
    trained_smoke_run, synthetic_image_dir
) -> None:
    """A single-image input produces a PNG mask and overlay of the same size."""
    config_path, _ = trained_smoke_run
    image_path = sorted(synthetic_image_dir.glob("*.png"))[0]

    result = _run_infer(config_path, image_path)
    assert result.returncode == 0, result.stderr

    config = yaml.safe_load(config_path.read_text())
    infer_dirs = sorted(
        (Path(config["outputs"]["checkpoints_dir"]).parent / "infer").glob(f"{config_path.stem}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    assert infer_dirs, "No infer output directory created"
    infer_dir = infer_dirs[-1]

    pred_path = infer_dir / "predictions" / f"{image_path.stem}.png"
    overlay_path = infer_dir / "overlays" / f"{image_path.stem}.png"
    assert pred_path.exists()
    assert overlay_path.exists()

    mask = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
    input_img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    assert mask.shape == input_img.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1, 2})


def test_infer_cli_processes_directory(trained_smoke_run, synthetic_image_dir) -> None:
    """A directory input processes every supported image file."""
    config_path, _ = trained_smoke_run

    result = _run_infer(config_path, synthetic_image_dir)
    assert result.returncode == 0, result.stderr

    config = yaml.safe_load(config_path.read_text())
    infer_dir = sorted(
        (Path(config["outputs"]["checkpoints_dir"]).parent / "infer").glob(f"{config_path.stem}_*"),
        key=lambda p: p.stat().st_mtime,
    )[-1]

    pred_files = {p.stem for p in (infer_dir / "predictions").glob("*.png")}
    input_files = {p.stem for p in synthetic_image_dir.glob("*.png")}
    assert pred_files == input_files


def test_infer_cli_is_deterministic(trained_smoke_run, synthetic_image_dir) -> None:
    """Two identical infer invocations produce pixel-identical masks."""
    config_path, _ = trained_smoke_run
    image_path = sorted(synthetic_image_dir.glob("*.png"))[0]

    result1 = _run_infer(config_path, image_path)
    result2 = _run_infer(config_path, image_path)
    assert result1.returncode == 0, result1.stderr
    assert result2.returncode == 0, result2.stderr

    config = yaml.safe_load(config_path.read_text())
    infer_dirs = sorted(
        (Path(config["outputs"]["checkpoints_dir"]).parent / "infer").glob(f"{config_path.stem}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    assert len(infer_dirs) >= 2
    dir1, dir2 = infer_dirs[-2], infer_dirs[-1]

    mask1 = cv2.imread(str(dir1 / "predictions" / f"{image_path.stem}.png"), cv2.IMREAD_UNCHANGED)
    mask2 = cv2.imread(str(dir2 / "predictions" / f"{image_path.stem}.png"), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(mask1, mask2)


def test_infer_cli_missing_input_exits_nonzero(trained_smoke_run, tmp_path: Path) -> None:
    """A missing --input path exits non-zero with a clear message."""
    config_path, _ = trained_smoke_run
    missing_path = tmp_path / "does_not_exist.png"
    result = _run_infer(config_path, missing_path)
    assert result.returncode != 0
    assert "input" in result.stderr.lower() or "exist" in result.stderr.lower()


def test_infer_cli_unresolvable_checkpoint_exits_nonzero(tmp_path: Path) -> None:
    """Infer exits non-zero when no checkpoint can be resolved."""
    config_path = _write_config(tmp_path, Path("configs/tiny.yaml"))
    image_path = tmp_path / "dummy.png"
    cv2.imwrite(str(image_path), np.zeros((64, 64), dtype=np.uint8))
    result = _run_infer(config_path, image_path, checkpoint="/does/not/exist.msgpack")
    assert result.returncode != 0
    assert "checkpoint" in result.stderr.lower()


def test_infer_cli_skips_unsupported_extensions_with_warning(
    trained_smoke_run, synthetic_image_dir, tmp_path: Path
) -> None:
    """Unsupported file extensions are skipped with a warning."""
    config_path, _ = trained_smoke_run
    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()
    for png_path in synthetic_image_dir.glob("*.png"):
        shutil.copy(png_path, mixed_dir / png_path.name)
    (mixed_dir / "notes.txt").write_text("not an image")

    result = _run_infer(config_path, mixed_dir)
    assert result.returncode == 0, result.stderr
    assert "notes" in result.stderr or "txt" in result.stderr or "warning" in result.stderr.lower()

    config = yaml.safe_load(config_path.read_text())
    infer_dir = sorted(
        (Path(config["outputs"]["checkpoints_dir"]).parent / "infer").glob(f"{config_path.stem}_*"),
        key=lambda p: p.stat().st_mtime,
    )[-1]
    pred_files = {p.name for p in (infer_dir / "predictions").glob("*.png")}
    assert "notes.txt" not in pred_files
    assert len(pred_files) == len(list(synthetic_image_dir.glob("*.png")))
