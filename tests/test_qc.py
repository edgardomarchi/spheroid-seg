"""Tests for annotation quality-control validation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile

from spheroid_seg.data.qc import run_qc


def test_qc_accepts_valid_fixture(
    tmp_raw_mask_dirs: tuple[Path, Path], make_pair: callable
) -> None:
    """QC passes when raw and mask pairs follow the annotation spec."""
    raw_dir, masks_dir = tmp_raw_mask_dirs
    make_pair("valid")

    exit_code = run_qc(raw_dir, masks_dir, output_dir=tmp_raw_mask_dirs[0].parent / "qc")
    assert exit_code == 0


def test_qc_rejects_shape_mismatch(
    tmp_raw_mask_dirs: tuple[Path, Path], make_pair: callable
) -> None:
    """QC fails when a raw image and its mask have different shapes."""
    raw_dir, masks_dir = tmp_raw_mask_dirs
    make_pair("bad_shape", shape=(128, 128))

    mask_path = masks_dir / "bad_shape.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(mask, (64, 128), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(mask_path), resized)

    exit_code = run_qc(raw_dir, masks_dir, output_dir=tmp_raw_mask_dirs[0].parent / "qc")
    assert exit_code == 1


def test_qc_rejects_wrong_dtype(
    tmp_raw_mask_dirs: tuple[Path, Path], make_pair: callable
) -> None:
    """QC fails when a mask is not uint8."""
    raw_dir, masks_dir = tmp_raw_mask_dirs
    make_pair("bad_dtype")

    mask_path = masks_dir / "bad_dtype.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    float_mask_path = masks_dir / "bad_dtype.tiff"
    tifffile.imwrite(float_mask_path, mask.astype(np.float32))
    mask_path.unlink()

    exit_code = run_qc(raw_dir, masks_dir, output_dir=tmp_raw_mask_dirs[0].parent / "qc")
    assert exit_code == 1


def test_qc_rejects_invalid_mask_value(
    tmp_raw_mask_dirs: tuple[Path, Path], make_pair: callable
) -> None:
    """QC fails when a mask contains a pixel value outside {0,1,2,3}."""
    raw_dir, masks_dir = tmp_raw_mask_dirs
    make_pair("bad_value")

    mask_path = masks_dir / "bad_value.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    mask[0, 0] = 4
    cv2.imwrite(str(mask_path), mask)

    exit_code = run_qc(raw_dir, masks_dir, output_dir=tmp_raw_mask_dirs[0].parent / "qc")
    assert exit_code == 1


def test_qc_graceful_when_no_pairs(tmp_path: Path) -> None:
    """QC exits cleanly with an informative message when no pairs are found."""
    raw_dir = tmp_path / "empty_raw"
    masks_dir = tmp_path / "empty_masks"
    raw_dir.mkdir()
    masks_dir.mkdir()

    exit_code = run_qc(raw_dir, masks_dir, output_dir=tmp_path / "qc")
    assert exit_code == 0
