"""End-to-end tests for the validation diagnostics CLI.

The suite trains a tiny model on a hermetic synthetic dataset, then exercises
the diagnostics command: output files, bias-0 equivalence with the existing
eval path, saved-patch prevalence, missing/malformed NPZ handling, argument
validation, --max-images, and run-to-run determinism.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from spheroid_seg.data.synthetic import generate_synthetic_dataset, synthetic_split_names


def _write_config(tmp_path: Path, raw_dir: Path, masks_dir: Path, splits_dir: Path) -> Path:
    """Copy tiny.yaml with repo-local data and output dirs."""
    config = yaml.safe_load(Path("configs/tiny.yaml").read_text())
    config["data"] = {
        "raw_dir": str(raw_dir),
        "masks_dir": str(masks_dir),
        "splits_dir": str(splits_dir),
        "slimia_dir": str(tmp_path / "slimia"),
    }
    config["outputs"] = {
        "checkpoints_dir": str(tmp_path / "outputs" / "checkpoints"),
        "logs_dir": str(tmp_path / "outputs" / "logs"),
        "predictions_dir": str(tmp_path / "outputs" / "predictions"),
        "metrics_dir": str(tmp_path / "outputs" / "metrics"),
        "qc_dir": str(tmp_path / "outputs" / "qc"),
        "debug_dir": str(tmp_path / "outputs" / "debug"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _run_train(config_path: Path, epochs: int = 2) -> Path:
    result = _run(
        [
            sys.executable,
            "-m",
            "spheroid_seg.train",
            "--config",
            str(config_path),
            "--epochs",
            str(epochs),
        ]
    )
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(config_path.read_text())
    runs_dir = Path(config["outputs"]["checkpoints_dir"]).parent / "runs"
    run_dirs = sorted(runs_dir.glob(f"{config_path.stem}_*"), key=lambda p: p.stat().st_mtime)
    assert run_dirs, f"No run directory found under {runs_dir}"
    return run_dirs[-1]


def _run_eval(config_path: Path, run_dir: Path) -> subprocess.CompletedProcess:
    return _run(
        [
            sys.executable,
            "-m",
            "spheroid_seg.eval",
            "--config",
            str(config_path),
            "--split",
            "val",
            "--run-dir",
            str(run_dir),
        ]
    )


def _run_diagnostics(config_path: Path, **kwargs) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "-m",
        "spheroid_seg.validation_diagnostics",
        "--config",
        str(config_path),
    ]
    for key, value in kwargs.items():
        cmd.append(f"--{key.replace('_', '-')}")
        if value is not True:
            cmd.append(str(value))
    return _run(cmd)


def _diagnostics_dirs(config_path: Path) -> list[Path]:
    config = yaml.safe_load(config_path.read_text())
    root = Path(config["outputs"]["checkpoints_dir"]).parent / "diagnostics"
    return sorted(root.glob(f"{config_path.stem}_*"), key=lambda p: p.stat().st_mtime)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Train a tiny model on a hermetic synthetic dataset (8 images, 256px)."""
    tmp_path = tmp_path_factory.mktemp("diagnostics_cli")
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    splits_dir = tmp_path / "splits"
    generate_synthetic_dataset(raw_dir, masks_dir, n_images=8, shape=(256, 256), seed=42)
    splits_dir.mkdir()
    for split, names in synthetic_split_names(8, 42).items():
        (splits_dir / f"{split}.txt").write_text("".join(f"{n}\n" for n in names))

    config_path = _write_config(tmp_path, raw_dir, masks_dir, splits_dir)
    run_dir = _run_train(config_path, epochs=2)
    return {"tmp_path": tmp_path, "config_path": config_path, "run_dir": run_dir}


def test_cli_smoke_writes_all_outputs(trained_run: dict) -> None:
    """A CPU smoke run writes every expected output file."""
    result = _run_diagnostics(
        trained_run["config_path"], run_dir=trained_run["run_dir"], split="val"
    )
    assert result.returncode == 0, result.stderr

    dirs = _diagnostics_dirs(trained_run["config_path"])
    assert dirs, "No diagnostics output directory created"
    out = dirs[-1]
    for name in (
        "patch_prevalence.csv",
        "fp_components.csv",
        "fp_component_summary.csv",
        "bias_sweep.csv",
        "bias_sweep_confusion.csv",
        "summary.md",
    ):
        assert (out / name).exists(), f"missing output: {name}"


def test_bias_zero_matches_existing_eval(trained_run: dict) -> None:
    """Bias 0 reproduces the existing eval confusion matrix and proper scores.

    Acceptance invariant 1/2 of the bias sweep: same checkpoint, config, and
    split must give the identical pooled 3x3 confusion matrix, and the proper
    scores must be exactly equal because the accumulation order matches the
    eval path tile for tile.
    """
    config_path, run_dir = trained_run["config_path"], trained_run["run_dir"]
    eval_result = _run_eval(config_path, run_dir)
    assert eval_result.returncode == 0, eval_result.stderr

    diag_result = _run_diagnostics(config_path, run_dir=run_dir, split="val")
    assert diag_result.returncode == 0, diag_result.stderr

    config = yaml.safe_load(config_path.read_text())
    evals_dir = Path(config["outputs"]["checkpoints_dir"]).parent / "evals"
    eval_dir = sorted(evals_dir.glob(f"{config_path.stem}_*"), key=lambda p: p.stat().st_mtime)[-1]
    out = _diagnostics_dirs(config_path)[-1]

    with (eval_dir / "confusion_matrix.csv").open("r", newline="") as f:
        eval_rows = list(csv.reader(f))
    eval_confusion = [[int(v) for v in row[1:]] for row in eval_rows[1:]]

    sweep_rows = _read_csv(out / "bias_sweep_confusion.csv")
    zero = {
        (r["gt"], r["prediction"]): int(r["count"])
        for r in sweep_rows
        if r["background_bias"] == "0.0" and r["group"] == "overall"
    }
    class_names = ["background", "loose cell", "aggregate"]
    diag_confusion = [[zero[(gt, pred)] for pred in class_names] for gt in class_names]
    assert diag_confusion == eval_confusion

    # Per-class Dice/IoU and proper scores must match exactly as well.
    with (eval_dir / "metrics.csv").open("r", newline="") as f:
        eval_metrics = {(r["group"], r["class"]): r for r in csv.DictReader(f)}
    diag_metrics = {
        (r["group"], r["class"]): r
        for r in _read_csv(out / "bias_sweep.csv")
        if r["background_bias"] == "0.0"
    }
    for cls in class_names:
        assert float(diag_metrics[("overall", cls)]["dice"]) == float(
            eval_metrics[("overall", cls)]["dice"]
        )
        assert float(diag_metrics[("overall", cls)]["iou"]) == float(
            eval_metrics[("overall", cls)]["iou"]
        )
        assert float(diag_metrics[("overall", cls)]["brier"]) == float(
            eval_metrics[("overall", cls)]["brier"]
        )
        assert float(diag_metrics[("overall", cls)]["log_loss"]) == float(
            eval_metrics[("overall", cls)]["log_loss"]
        )
    metrics = json.loads((eval_dir / "metrics.json").read_text())
    assert float(diag_metrics[("overall", "all")]["brier"]) == metrics["overall"]["brier_all"]
    assert float(diag_metrics[("overall", "all")]["log_loss"]) == metrics["overall"]["log_loss_all"]

    # Object rows exist with the same Dice as the eval object row.
    assert float(diag_metrics[("overall", "object")]["dice"]) == float(
        eval_metrics[("overall", "object")]["dice"]
    )


def test_saved_patch_prevalence_matches_npz(trained_run: dict) -> None:
    """Saved-patch rows in patch_prevalence.csv are exact NPZ mask counts."""
    config_path, run_dir = trained_run["config_path"], trained_run["run_dir"]
    result = _run_diagnostics(config_path, run_dir=run_dir, split="val")
    assert result.returncode == 0, result.stderr
    out = _diagnostics_dirs(config_path)[-1]

    npz = np.load(run_dir / "checkpoints" / "training_patches.npz")
    rows = _read_csv(out / "patch_prevalence.csv")
    for split in ("train", "val"):
        masks = npz[f"{split}_masks"]
        expected = np.bincount(masks.ravel().astype(np.int64), minlength=3)
        total = int(expected.sum())
        for cls_idx, cls in enumerate(("background", "loose cell", "aggregate")):
            row = next(
                r
                for r in rows
                if r["source"] == "saved_patch" and r["split"] == split and r["class_name"] == cls
            )
            assert int(row["pixel_count"]) == int(expected[cls_idx])
            assert float(row["pixel_fraction"]) == pytest.approx(
                expected[cls_idx] / total, rel=1e-12
            )

    # Fractions per (source, split) sum to 1.
    for source in ("full_image", "saved_patch"):
        for split in ("train", "val"):
            frac = sum(
                float(r["pixel_fraction"])
                for r in rows
                if r["source"] == source and r["split"] == split
            )
            assert frac == pytest.approx(1.0, rel=1e-9)

    # The summary reports the saved-patch / full-image foreground ratio.
    summary = (out / "summary.md").read_text()
    assert "foreground" in summary and "ratio" in summary


def test_missing_patches_npz_fails_clearly(trained_run: dict) -> None:
    """Without training_patches.npz the command fails unless skipping explicitly."""
    config_path, run_dir = trained_run["config_path"], trained_run["run_dir"]
    fake_run = trained_run["tmp_path"] / "fake_run"
    (fake_run / "checkpoints").mkdir(parents=True)
    shutil.copy(
        run_dir / "checkpoints" / "best_checkpoint.msgpack",
        fake_run / "checkpoints" / "best_checkpoint.msgpack",
    )

    result = _run_diagnostics(config_path, run_dir=fake_run, split="val")
    assert result.returncode != 0
    assert "training_patches.npz" in result.stderr

    skipped = _run_diagnostics(
        config_path, run_dir=fake_run, split="val", skip_saved_patch_prevalence=True
    )
    assert skipped.returncode == 0, skipped.stderr
    out = _diagnostics_dirs(config_path)[-1]
    rows = _read_csv(out / "patch_prevalence.csv")
    assert {r["source"] for r in rows} == {"full_image"}
    assert "skip" in (out / "summary.md").read_text().lower()


def test_malformed_patches_npz_fails_clearly(trained_run: dict) -> None:
    """An NPZ with an uninterpretable schema is rejected, not silently rebuilt."""
    config_path, run_dir = trained_run["config_path"], trained_run["run_dir"]
    fake_run = trained_run["tmp_path"] / "fake_run_malformed"
    (fake_run / "checkpoints").mkdir(parents=True)
    shutil.copy(
        run_dir / "checkpoints" / "best_checkpoint.msgpack",
        fake_run / "checkpoints" / "best_checkpoint.msgpack",
    )
    np.savez(
        fake_run / "checkpoints" / "training_patches.npz",
        train_images=np.zeros((2, 4, 4, 1), np.float32),
        train_masks=np.zeros((2, 4, 4), np.int32),
    )
    result = _run_diagnostics(config_path, run_dir=fake_run, split="val")
    assert result.returncode != 0
    assert "training_patches.npz" in result.stderr


@pytest.mark.parametrize(
    ("flag", "value", "hint"),
    [
        ("background_bias_grid", "0,0", "bias"),
        ("background_bias_grid", "nan", "bias"),
        ("area_thresholds", "0", "threshold"),
        ("area_thresholds", "16,16", "threshold"),
        ("max_images", "0", "max-images"),
    ],
)
def test_cli_rejects_invalid_arguments(trained_run: dict, flag: str, value: str, hint: str) -> None:
    """Invalid bias grids, area thresholds, and --max-images exit non-zero."""
    result = _run_diagnostics(
        trained_run["config_path"], run_dir=trained_run["run_dir"], split="val", **{flag: value}
    )
    assert result.returncode != 0
    assert hint in (result.stderr + result.stdout).lower()


def test_max_images_limits_the_split(trained_run: dict) -> None:
    """--max-images is a smoke/debug option that restricts the image count."""
    config_path, run_dir = trained_run["config_path"], trained_run["run_dir"]
    result = _run_diagnostics(
        config_path, run_dir=run_dir, split="val", max_images=1, background_bias_grid="0,1"
    )
    assert result.returncode == 0, result.stderr
    out = _diagnostics_dirs(config_path)[-1]

    rows = _read_csv(out / "bias_sweep.csv")
    assert {r["n_images"] for r in rows} == {"1"}
    component_rows = _read_csv(out / "fp_components.csv")
    assert len({r["image"] for r in component_rows}) <= 1


def test_cli_is_deterministic(trained_run: dict) -> None:
    """Two identical diagnostics runs produce byte-identical output files."""
    config_path, run_dir = trained_run["config_path"], trained_run["run_dir"]
    kwargs = {"run_dir": run_dir, "split": "val", "background_bias_grid": "0,0.5,1"}
    first = _run_diagnostics(config_path, **kwargs)
    second = _run_diagnostics(config_path, **kwargs)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    dirs = _diagnostics_dirs(config_path)
    dir1, dir2 = dirs[-2], dirs[-1]
    for name in (
        "patch_prevalence.csv",
        "fp_components.csv",
        "fp_component_summary.csv",
        "bias_sweep.csv",
        "bias_sweep_confusion.csv",
        "summary.md",
    ):
        assert (dir1 / name).read_bytes() == (dir2 / name).read_bytes(), (
            f"{name} differs between runs"
        )


def test_bias_sweep_monotonic_on_real_model(trained_run: dict) -> None:
    """On a trained checkpoint the foreground count never grows with the bias."""
    config_path, run_dir = trained_run["config_path"], trained_run["run_dir"]
    result = _run_diagnostics(config_path, run_dir=run_dir, split="val")
    assert result.returncode == 0, result.stderr
    out = _diagnostics_dirs(config_path)[-1]

    rows = [r for r in _read_csv(out / "bias_sweep.csv") if r["group"] == "overall"]
    by_bias: dict[float, int] = {}
    for r in rows:
        if r["class"] in ("loose cell", "aggregate"):
            bias = float(r["background_bias"])
            by_bias[bias] = by_bias.get(bias, 0) + int(r["pred_pixels"])
    biases = sorted(by_bias)
    assert biases == [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    counts = [by_bias[b] for b in biases]
    assert all(later <= earlier for earlier, later in zip(counts, counts[1:], strict=False))
    # The sweep strongly suppresses foreground; pixels whose float32
    # background probability underflowed to exactly 0 can never flip, so the
    # count need not reach zero on a real checkpoint (unit tests cover the
    # all-background guarantee with strictly positive probabilities).
    assert counts[-1] < counts[0]
