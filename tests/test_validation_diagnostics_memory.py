"""Peak-RSS regression test: diagnostics memory must scale sublinearly.

Like the eval memory guard, the validation diagnostics command streams
per-image work: it never retains per-pixel probability maps beyond the
current tile/batch and releases each image after folding it into small
accumulators. This test runs the real diagnostics CLI over 8 large images
and asserts peak RSS stays below 2x the peak of a 2-image run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from spheroid_seg.data.synthetic import write_synthetic_pair

# Same sizing rationale as tests/test_eval_memory.py: large enough that
# retaining whole-image probability maps across the loop would dominate
# process memory, small enough to keep the subprocess runs in seconds.
IMAGE_SHAPE = (5120, 5120)
N_IMAGES = 8
SMALL_N = 2

_RSS_WRAPPER = (
    "import resource, sys;"
    "from spheroid_seg.validation_diagnostics import main;"
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
) -> Path:
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
    config["eval"] = {"batch_size": 16, "num_overlay_samples": 4, "overlay_panel_width": 128}
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


@pytest.fixture(scope="module")
def memory_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """8 synthetic pairs plus disjoint train/val/test split files."""
    tmp_path = tmp_path_factory.mktemp("diagnostics_memory")
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    names = [f"mem{idx:02d}_{'4x' if idx % 2 == 0 else '10x'}" for idx in range(N_IMAGES)]
    for name in names:
        write_synthetic_pair(raw_dir, masks_dir, name, shape=IMAGE_SHAPE)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for split, members in (
        ("train", names[:4]),
        ("val", names[4:6]),
        ("test", names[6:]),
    ):
        (splits_dir / f"{split}.txt").write_text("".join(f"{n}\n" for n in members))

    return {
        "tmp_path": tmp_path,
        "raw_dir": raw_dir,
        "masks_dir": masks_dir,
        "names": names,
        "splits_dir": splits_dir,
    }


@pytest.fixture(scope="module")
def trained_run(memory_dataset: dict) -> Path:
    """Train one tiny epoch; both RSS runs reuse its checkpoint."""
    config_path = _write_config(
        memory_dataset["tmp_path"],
        "train_config",
        memory_dataset["raw_dir"],
        memory_dataset["masks_dir"],
        memory_dataset["splits_dir"],
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


def _run_diagnostics_peak_rss(config_path: Path, run_dir: Path) -> int:
    """Run diagnostics in a subprocess and return its peak RSS in kB."""
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
            "--background-bias-grid",
            "0,1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    peak_lines = [line for line in result.stdout.splitlines() if line.startswith("__PEAK_RSS_KB__")]
    assert peak_lines, f"peak RSS marker missing in stdout:\n{result.stdout}"
    return int(peak_lines[-1].split()[1])


def test_diagnostics_peak_rss_scales_sublinearly(memory_dataset: dict, trained_run: Path) -> None:
    """peak(8 images) < 2 * peak(2 images): no per-image probability retention."""
    names = memory_dataset["names"]
    tmp_path = memory_dataset["tmp_path"]

    def make_config(tag: str, val_names: list[str]) -> Path:
        splits_dir = tmp_path / f"splits_{tag}"
        splits_dir.mkdir(exist_ok=True)
        (splits_dir / "train.txt").write_text("".join(f"{n}\n" for n in names[:4]))
        (splits_dir / "val.txt").write_text("".join(f"{n}\n" for n in val_names))
        return _write_config(
            tmp_path,
            f"{tag}_config",
            memory_dataset["raw_dir"],
            memory_dataset["masks_dir"],
            splits_dir,
        )

    small_config = make_config("small", names[4 : 4 + SMALL_N])
    large_config = make_config("large", names)

    small_peak_kb = _run_diagnostics_peak_rss(small_config, trained_run)
    large_peak_kb = _run_diagnostics_peak_rss(large_config, trained_run)
    print(
        f"peak RSS: {SMALL_N} images -> {small_peak_kb / 1024:.0f} MB, "
        f"{N_IMAGES} images -> {large_peak_kb / 1024:.0f} MB "
        f"({large_peak_kb / small_peak_kb:.2f}x)"
    )

    assert large_peak_kb < 2 * small_peak_kb, (
        f"peak RSS scaled linearly: {SMALL_N} images -> {small_peak_kb} kB, "
        f"{N_IMAGES} images -> {large_peak_kb} kB "
        f"({large_peak_kb / small_peak_kb:.2f}x)"
    )
