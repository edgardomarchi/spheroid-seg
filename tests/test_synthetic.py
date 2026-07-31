"""Tests for synthetic dataset generation and deterministic splitting."""

from __future__ import annotations

from pathlib import Path

from spheroid_seg.data.synthetic import (
    generate_synthetic_dataset,
    synthetic_split_names,
)


def test_generate_synthetic_dataset_names_magnification(tmp_path: Path) -> None:
    """Synthetic files carry the 4x/10x suffix deterministically."""
    raw_dir, masks_dir = generate_synthetic_dataset(
        tmp_path / "raw",
        tmp_path / "masks",
        n_images=8,
        shape=(512, 512),
        seed=42,
    )

    names = sorted(p.stem for p in raw_dir.iterdir())
    assert names == [f"synth_{idx:03d}_{'4x' if idx % 2 == 0 else '10x'}" for idx in range(8)]
    assert all((masks_dir / f"{name}.png").exists() for name in names)


def test_synthetic_split_names_stratified() -> None:
    """The synthetic split contains both magnifications in each subset."""
    splits = synthetic_split_names(n_images=16, seed=42)

    assert set(splits.keys()) == {"train", "val", "test"}
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0

    for subset in splits.values():
        mags = {"4x" if name.endswith("_4x") else "10x" for name in subset}
        assert mags == {"4x", "10x"}


def test_synthetic_split_names_deterministic() -> None:
    """The synthetic split is deterministic for a fixed seed."""
    a = synthetic_split_names(n_images=16, seed=7)
    b = synthetic_split_names(n_images=16, seed=7)
    assert a == b


def test_synthetic_split_names_no_leak() -> None:
    """No image appears in more than one split."""
    splits = synthetic_split_names(n_images=16, seed=1)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set
