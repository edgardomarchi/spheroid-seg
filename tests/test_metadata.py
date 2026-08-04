"""Tests for magnification metadata parsing and CSV loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from spheroid_seg.data.metadata import load_metadata_csv, parse_magnification


def test_parse_magnification_4x() -> None:
    """Filenames ending in _4x return the '4x' group."""
    assert parse_magnification("img_001_4x.png") == "4x"
    assert parse_magnification("synth_000_4x") == "4x"


def test_parse_magnification_10x() -> None:
    """Filenames ending in _10x return the '10x' group."""
    assert parse_magnification("img_002_10x.png") == "10x"
    assert parse_magnification("synth_001_10x") == "10x"


def test_parse_magnification_unknown() -> None:
    """Filenames without a recognized suffix fall back to 'unknown'."""
    assert parse_magnification("img_003.png") == "unknown"
    assert parse_magnification("img_004_20x.png") == "unknown"
    assert parse_magnification("img_005_4x_extra.png") == "unknown"
    assert parse_magnification("") == "unknown"


def test_load_metadata_csv_precedence_over_filename(tmp_path: Path) -> None:
    """CSV entries override the filename-derived magnification.

    Keys are normalised to file stems so they match raw/mask base names
    regardless of extension.
    """
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text("image_name,magnification\nimg_4x.png,10x\nimg_plain.png,4x\n")

    meta = load_metadata_csv(csv_path)
    assert meta["img_4x"] == "10x"
    assert meta["img_plain"] == "4x"


def test_load_metadata_csv_malformed_header(tmp_path: Path) -> None:
    """A CSV missing the required columns raises a clear error."""
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text("name,mag\nimg.png,4x\n")

    with pytest.raises(ValueError, match="image_name"):
        load_metadata_csv(csv_path)


def test_load_metadata_csv_missing_image_row(tmp_path: Path) -> None:
    """A CSV row with an empty image name raises a clear error."""
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text("image_name,magnification\n,4x\n")

    with pytest.raises(ValueError, match="empty image_name"):
        load_metadata_csv(csv_path)


def test_load_metadata_csv_invalid_magnification(tmp_path: Path) -> None:
    """A CSV with an unsupported magnification value raises a clear error."""
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text("image_name,magnification\nimg.png,20x\n")

    with pytest.raises(ValueError, match="magnification"):
        load_metadata_csv(csv_path)
