"""Tests for train/val/test split file loading and leak detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from spheroid_seg.data.splits import check_split_leak, load_splits


def write_split(splits_dir: Path, name: str, names: list[str]) -> None:
    """Write a split text file with one base name per line."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / f"{name}.txt").write_text("\n".join(names) + "\n")


def test_load_splits_returns_ordered_names(tmp_path: Path) -> None:
    """Split files are read into ordered name lists."""
    splits_dir = tmp_path / "splits"
    write_split(splits_dir, "train", ["a", "b", "c"])
    write_split(splits_dir, "val", ["d"])
    write_split(splits_dir, "test", ["e", "f"])

    splits = load_splits(splits_dir)
    assert splits["train"] == ["a", "b", "c"]
    assert splits["val"] == ["d"]
    assert splits["test"] == ["e", "f"]


def test_load_splits_detects_leak(tmp_path: Path) -> None:
    """A clear error is raised when the same image appears in two splits."""
    splits_dir = tmp_path / "splits"
    write_split(splits_dir, "train", ["a", "b"])
    write_split(splits_dir, "val", ["b", "c"])
    write_split(splits_dir, "test", ["d"])

    with pytest.raises(ValueError, match="Patch-level leak detected"):
        load_splits(splits_dir)


def test_load_splits_does_not_shuffle(tmp_path: Path) -> None:
    """Split contents are never re-shuffled."""
    splits_dir = tmp_path / "splits"
    names = [f"img_{i:03d}" for i in range(20)]
    write_split(splits_dir, "train", names)
    write_split(splits_dir, "val", [])
    write_split(splits_dir, "test", [])

    splits = load_splits(splits_dir)
    assert splits["train"] == names


def test_check_split_leak_empty_succeeds() -> None:
    """Leak detection succeeds on empty split dictionaries."""
    check_split_leak({})
