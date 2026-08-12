"""End-to-end TIFF support tests (LZW compression)."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import tifffile

from spheroid_seg.data.dataset import load_pair
from spheroid_seg.data.make_metadata import main as make_metadata_main
from spheroid_seg.data.make_splits import main as make_splits_main
from spheroid_seg.data.qc import run_qc


def _write_png(path: Path, array: np.ndarray) -> None:
    """Write a grayscale image/mask as PNG."""
    cv2.imwrite(str(path), array)


def _write_lzw_tiff(path: Path, array: np.ndarray) -> None:
    """Write an array as an LZW-compressed TIFF."""
    tifffile.imwrite(path, array, compression="lzw")


def _make_config(tmp_path: Path, raw_dir: Path, masks_dir: Path, splits_dir: Path) -> Path:
    """Create a minimal YAML config for split generation."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"seed: 42\n"
        f"batch_size: 2\n"
        f"data:\n"
        f"  raw_dir: {raw_dir}\n"
        f"  masks_dir: {masks_dir}\n"
        f"  splits_dir: {splits_dir}\n"
    )
    return config


def test_qc_accepts_lzw_tiff_pairs(tmp_path: Path) -> None:
    """QC passes on raw/mask pairs where both files are LZW TIFFs."""
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    raw_dir.mkdir()
    masks_dir.mkdir()

    raw = np.random.default_rng(42).integers(0, 256, size=(32, 32), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 2  # aggregate class

    _write_lzw_tiff(raw_dir / "sample_4x.tif", raw)
    _write_lzw_tiff(masks_dir / "sample_4x.tif", mask)

    assert run_qc(raw_dir, masks_dir, verbose=False) == 0


def test_dataset_loads_tiff_identically_to_png(tmp_path: Path) -> None:
    """load_pair returns the same arrays for a PNG and an LZW TIFF of the same data."""
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    raw_dir.mkdir()
    masks_dir.mkdir()

    rng = np.random.default_rng(7)
    raw = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[20:30, 20:30] = 2

    _write_png(raw_dir / "png_4x.png", raw)
    _write_png(masks_dir / "png_4x.png", mask)
    _write_lzw_tiff(raw_dir / "tiff_4x.tif", raw)
    _write_lzw_tiff(masks_dir / "tiff_4x.tif", mask)

    image_png, mask_png = load_pair(
        raw_dir / "png_4x.png",
        masks_dir / "png_4x.png",
        input_channels="grayscale",
        class_mapping={0: 0, 1: 1, 2: 2, 3: 2},
    )
    image_tiff, mask_tiff = load_pair(
        raw_dir / "tiff_4x.tif",
        masks_dir / "tiff_4x.tif",
        input_channels="grayscale",
        class_mapping={0: 0, 1: 1, 2: 2, 3: 2},
    )

    np.testing.assert_array_equal(image_png, image_tiff)
    np.testing.assert_array_equal(mask_png, mask_tiff)


def test_make_metadata_discovers_tiff_suffixes(tmp_path: Path) -> None:
    """make_metadata.py parses _4x/_10x from .tif and .tiff filenames."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "a_4x.tif").write_text("")
    (raw_dir / "b_10x.tiff").write_text("")
    (raw_dir / "c_4x.PNG").write_text("")

    config = tmp_path / "config.yaml"
    config.write_text(f"data:\n  raw_dir: {raw_dir}\n")

    assert make_metadata_main(["--config", str(config)]) == 0

    csv_path = raw_dir.parent / "metadata.csv"
    rows = list(csv.DictReader(csv_path.open("r", newline="")))
    by_name = {row["image"]: row for row in rows}
    assert by_name["a_4x.tif"]["magnification"] == "4x"
    assert by_name["b_10x.tiff"]["magnification"] == "10x"
    assert by_name["c_4x.PNG"]["magnification"] == "4x"


def test_mixed_png_tiff_directory_in_qc_and_splits(tmp_path: Path) -> None:
    """A directory with PNG raw and TIFF mask (and vice versa) is handled consistently."""
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    splits_dir = tmp_path / "splits"
    raw_dir.mkdir()
    masks_dir.mkdir()
    splits_dir.mkdir()

    rng = np.random.default_rng(3)
    for name, ext in (("png_raw_4x", ".png"), ("tiff_raw_10x", ".tiff")):
        raw = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 2
        if ext == ".png":
            _write_png(raw_dir / f"{name}.png", raw)
            _write_lzw_tiff(masks_dir / f"{name}.tif", mask)
        else:
            _write_lzw_tiff(raw_dir / f"{name}.tif", raw)
            _write_png(masks_dir / f"{name}.png", mask)

    assert run_qc(raw_dir, masks_dir, verbose=False) == 0

    metadata_path = raw_dir.parent / "metadata.csv"
    metadata_path.write_text(
        "image,magnification,condition\npng_raw_4x.png,4x,\ntiff_raw_10x.tif,10x,\n"
    )

    config = _make_config(tmp_path, raw_dir, masks_dir, splits_dir)
    assert make_splits_main(["--config", str(config), "--metadata", str(metadata_path)]) == 0

    for split in ("train", "val", "test"):
        assert (splits_dir / f"{split}.txt").exists()


def test_qc_rejects_tiff_mask_with_out_of_range_values(tmp_path: Path) -> None:
    """A TIFF mask containing a label outside {0,1,2,3} is flagged like a PNG one."""
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    raw_dir.mkdir()
    masks_dir.mkdir()

    raw = np.full((16, 16), 128, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4, 4] = 5

    _write_lzw_tiff(raw_dir / "bad_4x.tif", raw)
    _write_lzw_tiff(masks_dir / "bad_4x.tif", mask)

    assert run_qc(raw_dir, masks_dir, verbose=False) == 1


def test_infer_preprocesses_tiff(tmp_path: Path) -> None:
    """The inference preprocessor can read LZW TIFF raw images."""
    from spheroid_seg.infer import _preprocess_image

    image_path = tmp_path / "raw_4x.tif"
    raw = np.random.default_rng(11).integers(0, 256, size=(32, 32), dtype=np.uint8)
    _write_lzw_tiff(image_path, raw)

    processed = _preprocess_image(image_path, "grayscale")
    assert processed.dtype == np.float32
    assert processed.ndim == 2
    assert 0 <= processed.min() <= processed.max() <= 1
