"""Peak-RSS regression test: eval memory must scale sublinearly with image count.

Evaluation streams per-image work and only retains small aggregated state
(pooled confusion counts, per-image scalar metrics, and the few overlay panels
selected for the grid). This test guards that property: running the real eval
CLI over 8 images must not use twice the peak RSS of a run over 2 images.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from spheroid_seg.data.synthetic import write_synthetic_pair

# Large enough that retaining whole images across the eval loop (the old,
# pre-streaming behavior) dominates process baseline memory; small enough to
# keep the two eval subprocess runs in seconds. At this size the pre-streaming
# pipeline retains ~0.35 GB per image, which pushes an 8-image run past 2x the
# peak RSS of a 2-image run (measured 2.0x-2.3x before the streaming fix).
IMAGE_SHAPE = (5120, 5120)
N_IMAGES = 8
SMALL_N = 2

# Runs eval in-process and prints its own peak RSS so the parent can measure
# the eval process exactly (no shell, no GNU time dependency).
_RSS_WRAPPER = (
    "import resource, sys;"
    "from spheroid_seg.eval import main;"
    "rc = main(sys.argv[1:]);"
    "print(f'__PEAK_RSS_KB__ {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}',"
    " flush=True);"
    "sys.exit(rc)"
)


def _write_config(
    tmp_path: Path,
    name: str,
    raw_dir: Path,
    masks_dir: Path,
    splits_dir: Path,
    overrides: dict | None = None,
) -> Path:
    """Copy tiny.yaml into a temp dir with repo-local data/output dirs."""
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
    if overrides:
        config.update(overrides)
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


@pytest.fixture(scope="module")
def memory_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """8 synthetic pairs plus disjoint train/val/test split files."""
    tmp_path = tmp_path_factory.mktemp("eval_memory")
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    names = [f"mem{idx:02d}_{'4x' if idx % 2 == 0 else '10x'}" for idx in range(N_IMAGES)]
    for name in names:
        write_synthetic_pair(raw_dir, masks_dir, name, shape=IMAGE_SHAPE)

    train_splits = tmp_path / "splits_train"
    train_splits.mkdir()
    for split, members in (
        ("train", names[:4]),
        ("val", names[4:6]),
        ("test", names[6:]),
    ):
        (train_splits / f"{split}.txt").write_text("".join(f"{n}\n" for n in members))

    return {
        "tmp_path": tmp_path,
        "raw_dir": raw_dir,
        "masks_dir": masks_dir,
        "names": names,
        "train_splits": train_splits,
    }


@pytest.fixture(scope="module")
def trained_run(memory_dataset: dict) -> Path:
    """Train one tiny epoch on the dataset; both RSS runs reuse its checkpoint."""
    config_path = _write_config(
        memory_dataset["tmp_path"],
        "train_config",
        memory_dataset["raw_dir"],
        memory_dataset["masks_dir"],
        memory_dataset["train_splits"],
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
    assert result.returncode == 0, result.stderr
    runs_dir = memory_dataset["tmp_path"] / "outputs" / "runs"
    run_dirs = sorted(runs_dir.glob("train_config_*"), key=lambda p: p.stat().st_mtime)
    assert run_dirs, f"No run directory found under {runs_dir}"
    return run_dirs[-1]


def _run_eval_peak_rss(config_path: Path, run_dir: Path) -> tuple[int, Path]:
    """Run eval in a subprocess and return (peak RSS kB, eval output dir)."""
    evals_dir = config_path.parent / "outputs" / "evals"
    before = sorted(evals_dir.glob("*")) if evals_dir.exists() else []
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _RSS_WRAPPER,
            "--config",
            str(config_path),
            "--split",
            "val",
            "--run-dir",
            str(run_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    peak_lines = [line for line in result.stdout.splitlines() if line.startswith("__PEAK_RSS_KB__")]
    assert peak_lines, f"peak RSS marker missing in stdout:\n{result.stdout}"
    peak_kb = int(peak_lines[-1].split()[1])

    evals_dir = config_path.parent / "outputs" / "evals"
    eval_dirs = sorted(evals_dir.glob(f"{config_path.stem}_*"), key=lambda p: p.stat().st_mtime)
    new_dirs = [d for d in eval_dirs if d not in before]
    assert new_dirs, f"No new eval output directory in {evals_dir}"
    return peak_kb, new_dirs[-1]


def test_eval_peak_rss_scales_sublinearly(memory_dataset: dict, trained_run: Path) -> None:
    """peak(8 images) < 2 * peak(2 images): eval streams, never retains all images."""
    names = memory_dataset["names"]
    tmp_path = memory_dataset["tmp_path"]

    def make_run_config(tag: str, val_names: list[str]) -> Path:
        splits_dir = tmp_path / f"splits_{tag}"
        splits_dir.mkdir(exist_ok=True)
        (splits_dir / "val.txt").write_text("".join(f"{n}\n" for n in val_names))
        # Larger eval batch keeps the subprocess runtime in seconds.
        return _write_config(
            tmp_path,
            f"{tag}_config",
            memory_dataset["raw_dir"],
            memory_dataset["masks_dir"],
            splits_dir,
            overrides={
                "eval": {
                    "batch_size": 16,
                    "num_overlay_samples": 4,
                    "overlay_panel_width": 128,
                }
            },
        )

    small_config = make_run_config("small", names[4 : 4 + SMALL_N])
    large_config = make_run_config("large", names)

    small_peak_kb, small_eval_dir = _run_eval_peak_rss(small_config, trained_run)
    large_peak_kb, large_eval_dir = _run_eval_peak_rss(large_config, trained_run)
    print(
        f"peak RSS: {SMALL_N} images -> {small_peak_kb / 1024:.0f} MB, "
        f"{N_IMAGES} images -> {large_peak_kb / 1024:.0f} MB "
        f"({large_peak_kb / small_peak_kb:.2f}x)"
    )

    # Both runs must have produced a real metrics report.
    for eval_dir in (small_eval_dir, large_eval_dir):
        metrics = json.loads((eval_dir / "metrics.json").read_text())
        assert metrics["per_image"], f"empty per-image metrics in {eval_dir}"

    # Guard: 4x the images must not double peak memory. The pre-streaming
    # implementation retained every image, mask, prediction, and overlay
    # buffer, growing peak RSS linearly with the image count.
    assert large_peak_kb < 2 * small_peak_kb, (
        f"peak RSS scaled linearly: {SMALL_N} images -> {small_peak_kb} kB, "
        f"{N_IMAGES} images -> {large_peak_kb} kB "
        f"({large_peak_kb / small_peak_kb:.2f}x)"
    )
