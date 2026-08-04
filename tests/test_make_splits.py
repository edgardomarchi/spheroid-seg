"""Tests for scripts/make_splits.py real-data split creation."""

from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from spheroid_seg.data.splits import load_splits
from spheroid_seg.data.synthetic import write_synthetic_pair

REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "scripts" / "make_splits.py"


def _write_config(
    tmp_path: Path,
    *,
    seed: int = 42,
    raw_dir: Path | None = None,
    masks_dir: Path | None = None,
    splits_dir: Path | None = None,
) -> Path:
    """Write a minimal config file pointing data paths into tmp_path."""
    config: dict[str, Any] = {
        "seed": seed,
        "num_classes": 3,
        "patch_size": 128,
        "base_features": 16,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "batch_size": 2,
        "epochs": 2,
        "early_stopping_patience": 2,
        "class_weights": [0.5, 1.0, 1.0],
        "input_channels": "grayscale",
        "class_mapping": {0: 0, 1: 1, 2: 2, 3: 2},
        "min_object_fraction": 0.05,
        "object_patch_ratio": 0.8,
        "patches_per_image": 2,
        "augment": {
            "flip_probability": 0.5,
            "rotate90_probability": 0.5,
            "elastic_probability": 0.0,
            "scale_probability": 0.0,
            "blur_probability": 0.0,
            "noise_probability": 0.0,
            "brightness_contrast_probability": 0.0,
            "scale_range": [0.5, 2.0],
            "elastic_alpha": 1,
            "elastic_sigma": 50,
            "blur_limit": [3, 7],
            "blur_sigma_limit": [0.1, 2.0],
            "noise_std_range": [0.01, 0.05],
            "brightness_limit": 0.3,
            "contrast_limit": 0.3,
        },
        "data": {
            "raw_dir": str(raw_dir or tmp_path / "raw"),
            "masks_dir": str(masks_dir or tmp_path / "masks"),
            "splits_dir": str(splits_dir or tmp_path / "splits"),
            "slimia_dir": str(tmp_path / "slimia"),
        },
        "outputs": {
            "checkpoints_dir": str(tmp_path / "checkpoints"),
            "logs_dir": str(tmp_path / "logs"),
            "predictions_dir": str(tmp_path / "predictions"),
            "metrics_dir": str(tmp_path / "metrics"),
            "qc_dir": str(tmp_path / "qc"),
            "debug_dir": str(tmp_path / "debug"),
        },
        "eval": {"batch_size": 2, "num_overlay_samples": 2, "overlay_panel_width": 128},
        "infer": {"overlap": 0.15, "batch_size": 2},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _write_metadata(
    tmp_path: Path,
    rows: list[dict[str, str]],
    condition: bool = False,
) -> Path:
    """Write a metadata CSV with columns image,magnification[,condition]."""
    csv_path = tmp_path / "metadata.csv"
    header = "image,magnification" + (",condition" if condition else "") + "\n"
    lines = [header]
    for row in rows:
        line = f"{row['image']},{row['magnification']}"
        if condition:
            line += f",{row['condition']}"
        lines.append(line + "\n")
    csv_path.write_text("".join(lines))
    return csv_path


def _make_pairs(
    tmp_path: Path,
    stems: list[str],
) -> tuple[Path, Path]:
    """Create raw/mask pairs for the given file-name stems."""
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    for stem in stems:
        write_synthetic_pair(raw_dir, masks_dir, stem, shape=(64, 64))
    return raw_dir, masks_dir


def _run(
    tmp_path: Path,
    *args: str,
    expect_success: bool = True,
) -> subprocess.CompletedProcess:
    """Invoke scripts/make_splits.py with the provided arguments."""
    cmd = ["python", str(SCRIPT), *args]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if expect_success:
        assert result.returncode == 0, (
            f"Command failed: {' '.join(cmd)}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


@pytest.fixture
def balanced_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, list[str]]:
    """20 images per magnification (40 total), no condition column."""
    stems = [f"img_{i:03d}_{'4x' if i < 20 else '10x'}" for i in range(40)]
    raw_dir, masks_dir = _make_pairs(tmp_path, stems)
    metadata_rows = [
        {"image": f"{stem}.png", "magnification": stem.split("_")[-1]} for stem in stems
    ]
    metadata_path = _write_metadata(tmp_path, metadata_rows)
    config_path = _write_config(tmp_path, raw_dir=raw_dir, masks_dir=masks_dir)
    splits_dir = Path(config_path.parent) / "splits"
    return config_path, metadata_path, raw_dir, splits_dir, stems


def test_proportions_sum_to_group_size(balanced_fixture: tuple) -> None:
    """Each magnification is split ~70/15/15 and counts sum to the group size."""
    config_path, metadata_path, _raw_dir, splits_dir, names = balanced_fixture
    _run(config_path.parent, "--config", str(config_path), "--metadata", str(metadata_path))

    mag_counts: dict[str, Counter[str]] = {"4x": Counter(), "10x": Counter()}
    for name in names:
        mag = "4x" if name.endswith("_4x") else "10x"
        for split_name in ("train", "val", "test"):
            if name in (splits_dir / f"{split_name}.txt").read_text():
                mag_counts[mag][split_name] += 1

    for mag, counts in mag_counts.items():
        total = sum(counts.values())
        assert total == 20, f"{mag} group size should be 20, got {total}"
        # 70/15/15 with rounding: 20 -> 14/3/3
        assert counts["train"] == 14, f"{mag} train count should be 14, got {counts['train']}"
        assert counts["val"] == 3, f"{mag} val count should be 3, got {counts['val']}"
        assert counts["test"] == 3, f"{mag} test count should be 3, got {counts['test']}"


def test_joint_stratification_all_combinations_present(tmp_path: Path) -> None:
    """With enough samples per (magnification, condition) group, every split has every combo."""
    # 6 images per (mag, condition) combo -> 24 total
    stems = []
    rows = []
    for mag in ("4x", "10x"):
        for cond in ("A", "B"):
            for i in range(6):
                stem = f"{mag}_{cond}_{i:02d}"
                stems.append(stem)
                rows.append({"image": f"{stem}.png", "magnification": mag, "condition": cond})
    raw_dir, masks_dir = _make_pairs(tmp_path, stems)
    metadata_path = _write_metadata(tmp_path, rows, condition=True)
    config_path = _write_config(tmp_path, raw_dir=raw_dir, masks_dir=masks_dir)

    _run(tmp_path, "--config", str(config_path), "--metadata", str(metadata_path))
    splits_dir = tmp_path / "splits"
    for split_name in ("train", "val", "test"):
        split_names = {
            line.strip()
            for line in (splits_dir / f"{split_name}.txt").read_text().splitlines()
            if line.strip()
        }
        present_combos = set()
        for name in split_names:
            parts = name.split("_")
            present_combos.add((parts[0], parts[1]))
        assert present_combos == {("4x", "A"), ("4x", "B"), ("10x", "A"), ("10x", "B")}, (
            f"Split {split_name} missing combos: {present_combos}"
        )


def test_fallback_to_magnification_only_warns(tmp_path: Path) -> None:
    """A tiny (mag, condition) group falls back to magnification-only stratification."""
    stems = []
    rows = []
    # 4x/A has 6, 4x/B has 2 (too small), 10x/A has 6, 10x/B has 6
    for mag, cond, count in (("4x", "A", 6), ("4x", "B", 2), ("10x", "A", 6), ("10x", "B", 6)):
        for i in range(count):
            stem = f"{mag}_{cond}_{i:02d}"
            stems.append(stem)
            rows.append({"image": f"{stem}.png", "magnification": mag, "condition": cond})
    raw_dir, masks_dir = _make_pairs(tmp_path, stems)
    metadata_path = _write_metadata(tmp_path, rows, condition=True)
    config_path = _write_config(tmp_path, raw_dir=raw_dir, masks_dir=masks_dir)

    result = _run(tmp_path, "--config", str(config_path), "--metadata", str(metadata_path))
    assert (
        "falling back to magnification-only stratification"
        in (result.stdout + result.stderr).lower()
    )

    # Verify split files were still written and are valid.
    splits_dir = tmp_path / "splits"
    splits = load_splits(splits_dir)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(stems)


def test_determinism_same_seed_byte_identical(balanced_fixture: tuple) -> None:
    """Two runs with the same seed produce byte-identical split files."""
    config_path, metadata_path, _raw_dir, splits_dir, _names = balanced_fixture
    _run(config_path.parent, "--config", str(config_path), "--metadata", str(metadata_path))
    first_contents = {
        name: (splits_dir / f"{name}.txt").read_bytes() for name in ("train", "val", "test")
    }

    for name in ("train", "val", "test"):
        (splits_dir / f"{name}.txt").unlink()

    _run(config_path.parent, "--config", str(config_path), "--metadata", str(metadata_path))
    second_contents = {
        name: (splits_dir / f"{name}.txt").read_bytes() for name in ("train", "val", "test")
    }

    assert first_contents == second_contents


def test_different_seeds_different_assignments(balanced_fixture: tuple) -> None:
    """Different seeds produce different assignments with high probability."""
    config_path, metadata_path, _raw_dir, splits_dir, _names = balanced_fixture
    _run(config_path.parent, "--config", str(config_path), "--metadata", str(metadata_path))
    first_train = set((splits_dir / "train.txt").read_text().splitlines())

    for name in ("train", "val", "test"):
        (splits_dir / f"{name}.txt").unlink()

    config_path2 = _write_config(
        config_path.parent,
        seed=123,
        raw_dir=config_path.parent / "raw",
        masks_dir=config_path.parent / "masks",
    )
    _run(config_path.parent, "--config", str(config_path2), "--metadata", str(metadata_path))
    second_train = set((splits_dir / "train.txt").read_text().splitlines())

    assert first_train != second_train


def test_overwrite_safety_existing_files(tmp_path: Path) -> None:
    """Existing split files cause non-zero exit unless --force is passed."""
    stems = [f"img_{i:03d}_4x" for i in range(10)]
    raw_dir, masks_dir = _make_pairs(tmp_path, stems)
    metadata_path = _write_metadata(
        tmp_path,
        [{"image": f"{stem}.png", "magnification": "4x"} for stem in stems],
    )
    config_path = _write_config(tmp_path, raw_dir=raw_dir, masks_dir=masks_dir)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        (splits_dir / f"{name}.txt").write_text("placeholder\n")

    result = _run(
        tmp_path,
        "--config",
        str(config_path),
        "--metadata",
        str(metadata_path),
        expect_success=False,
    )
    assert result.returncode != 0
    assert "already exist" in (result.stdout + result.stderr).lower()

    result = _run(
        tmp_path, "--config", str(config_path), "--metadata", str(metadata_path), "--force"
    )
    assert result.returncode == 0
    # Files should now contain real names, not placeholder.
    assert "placeholder" not in (splits_dir / "train.txt").read_text()


def test_validation_metadata_without_files(tmp_path: Path) -> None:
    """A metadata row missing both raw and mask files is reported and exits non-zero."""
    stems = [f"img_{i:03d}_4x" for i in range(5)]
    raw_dir, masks_dir = _make_pairs(tmp_path, stems)
    rows = [{"image": f"{stem}.png", "magnification": "4x"} for stem in stems]
    rows.append({"image": "missing_4x.png", "magnification": "4x"})
    metadata_path = _write_metadata(tmp_path, rows)
    config_path = _write_config(tmp_path, raw_dir=raw_dir, masks_dir=masks_dir)

    result = _run(
        tmp_path,
        "--config",
        str(config_path),
        "--metadata",
        str(metadata_path),
        expect_success=False,
    )
    assert result.returncode != 0
    assert "missing_4x" in (result.stdout + result.stderr)


def test_validation_raw_without_mask(tmp_path: Path) -> None:
    """A raw file without a matching mask is reported and exits non-zero."""
    stems = [f"img_{i:03d}_4x" for i in range(5)]
    raw_dir, masks_dir = _make_pairs(tmp_path, stems)
    # Add an extra raw file with no mask.
    write_synthetic_pair(raw_dir, masks_dir, "orphan_raw_4x", shape=(64, 64))
    (masks_dir / "orphan_raw_4x.png").unlink()
    rows = [{"image": f"{stem}.png", "magnification": "4x"} for stem in stems]
    metadata_path = _write_metadata(tmp_path, rows)
    config_path = _write_config(tmp_path, raw_dir=raw_dir, masks_dir=masks_dir)

    result = _run(
        tmp_path,
        "--config",
        str(config_path),
        "--metadata",
        str(metadata_path),
        expect_success=False,
    )
    assert result.returncode != 0
    assert "orphan" in (result.stdout + result.stderr).lower()


def test_validation_mask_without_raw(tmp_path: Path) -> None:
    """A mask file without a matching raw file is reported and exits non-zero."""
    stems = [f"img_{i:03d}_4x" for i in range(5)]
    raw_dir, masks_dir = _make_pairs(tmp_path, stems)
    # Add an extra mask file with no raw.
    write_synthetic_pair(raw_dir, masks_dir, "orphan_mask_4x", shape=(64, 64))
    (raw_dir / "orphan_mask_4x.png").unlink()
    rows = [{"image": f"{stem}.png", "magnification": "4x"} for stem in stems]
    metadata_path = _write_metadata(tmp_path, rows)
    config_path = _write_config(tmp_path, raw_dir=raw_dir, masks_dir=masks_dir)

    result = _run(
        tmp_path,
        "--config",
        str(config_path),
        "--metadata",
        str(metadata_path),
        expect_success=False,
    )
    assert result.returncode != 0
    assert "orphan" in (result.stdout + result.stderr).lower()


def test_integration_loads_via_splits_py_and_passes_leak_check(balanced_fixture: tuple) -> None:
    """The written split files load through splits.py and pass the leak check."""
    config_path, metadata_path, _raw_dir, splits_dir, _names = balanced_fixture
    _run(config_path.parent, "--config", str(config_path), "--metadata", str(metadata_path))

    splits = load_splits(splits_dir)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0


def test_fraction_validation_out_of_range(tmp_path: Path) -> None:
    """Fractions outside (0, 1) are rejected."""
    stems = [f"img_{i:03d}_4x" for i in range(10)]
    raw_dir, masks_dir = _make_pairs(tmp_path, stems)
    metadata_path = _write_metadata(
        tmp_path, [{"image": f"{stem}.png", "magnification": "4x"} for stem in stems]
    )
    config_path = _write_config(tmp_path, raw_dir=raw_dir, masks_dir=masks_dir)

    result = _run(
        tmp_path,
        "--config",
        str(config_path),
        "--metadata",
        str(metadata_path),
        "--train-frac",
        "0",
        expect_success=False,
    )
    assert result.returncode != 0
    assert "train-frac" in (result.stdout + result.stderr).lower()

    result = _run(
        tmp_path,
        "--config",
        str(config_path),
        "--metadata",
        str(metadata_path),
        "--val-frac",
        "1.0",
        expect_success=False,
    )
    assert result.returncode != 0
    assert "val-frac" in (result.stdout + result.stderr).lower()


def test_fraction_validation_sum_too_large(tmp_path: Path) -> None:
    """Fractions summing to >= 1 are rejected."""
    stems = [f"img_{i:03d}_4x" for i in range(10)]
    raw_dir, masks_dir = _make_pairs(tmp_path, stems)
    metadata_path = _write_metadata(
        tmp_path, [{"image": f"{stem}.png", "magnification": "4x"} for stem in stems]
    )
    config_path = _write_config(tmp_path, raw_dir=raw_dir, masks_dir=masks_dir)

    result = _run(
        tmp_path,
        "--config",
        str(config_path),
        "--metadata",
        str(metadata_path),
        "--train-frac",
        "0.6",
        "--val-frac",
        "0.4",
        expect_success=False,
    )
    assert result.returncode != 0
    assert "sum" in (result.stdout + result.stderr).lower()
