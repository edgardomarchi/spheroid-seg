"""Tests for scripts/make_metadata.py and spheroid_seg.data.make_metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from spheroid_seg.data.make_metadata import main


def _make_config(tmp_path: Path, raw_dir: Path) -> Path:
    """Create a minimal YAML config pointing at the temp raw directory."""
    config = tmp_path / "config.yaml"
    config.write_text(f"data:\n  raw_dir: {raw_dir}\n")
    return config


def _read_csv(csv_path: Path) -> list[str]:
    """Return the CSV lines, stripping trailing newlines."""
    return csv_path.read_text().strip("\n").split("\n")


def test_parses_magnification_from_names_with_spaces(tmp_path: Path, capsys) -> None:
    """Spaces in filenames are preserved and the trailing _4x/_10x suffix is parsed."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "117 - 9d 3T3_10x.png").write_text("")
    (raw_dir / "216 - 3d 3T3_4x.png").write_text("")

    config = _make_config(tmp_path, raw_dir)
    assert main(["--config", str(config)]) == 0

    csv_path = raw_dir.parent / "metadata.csv"
    lines = _read_csv(csv_path)
    assert lines == [
        "image,magnification,condition",
        "117 - 9d 3T3_10x.png,10x,",
        "216 - 3d 3T3_4x.png,4x,",
    ]


def test_recognizes_suffix_with_uppercase_extension(tmp_path: Path) -> None:
    """Extension case does not prevent suffix parsing."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sample_4x.PNG").write_text("")
    (raw_dir / "sample_10x.JPG").write_text("")

    config = _make_config(tmp_path, raw_dir)
    assert main(["--config", str(config)]) == 0

    csv_path = raw_dir.parent / "metadata.csv"
    text = csv_path.read_text()
    assert "sample_4x.PNG,4x," in text
    assert "sample_10x.JPG,10x," in text


def test_files_without_suffix_get_empty_magnification_and_warning(tmp_path: Path, capsys) -> None:
    """Unrecognized filenames are kept but flagged for manual magnification."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "needs_manual.png").write_text("")

    config = _make_config(tmp_path, raw_dir)
    assert main(["--config", str(config)]) == 0

    csv_path = raw_dir.parent / "metadata.csv"
    lines = _read_csv(csv_path)
    assert lines == [
        "image,magnification,condition",
        "needs_manual.png,,",
    ]

    out, err = capsys.readouterr()
    assert "1 file(s) need manual magnification" in err
    assert "needs_manual.png" in err


def test_deterministic_alphabetical_order(tmp_path: Path) -> None:
    """Rows are written sorted by filename regardless of creation order."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in ("z_10x.png", "a_4x.png", "m_10x.png"):
        (raw_dir / name).write_text("")

    config = _make_config(tmp_path, raw_dir)
    assert main(["--config", str(config)]) == 0

    csv_path = raw_dir.parent / "metadata.csv"
    rows = _read_csv(csv_path)[1:]
    assert rows == ["a_4x.png,4x,", "m_10x.png,10x,", "z_10x.png,10x,"]


def test_refuses_to_overwrite_without_force(tmp_path: Path, capsys) -> None:
    """A second run without --update or --force fails and leaves the CSV intact."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "img_4x.png").write_text("")

    config = _make_config(tmp_path, raw_dir)
    assert main(["--config", str(config)]) == 0

    csv_path = raw_dir.parent / "metadata.csv"
    original = csv_path.read_text()

    assert main(["--config", str(config)]) == 1
    out, err = capsys.readouterr()
    assert "already exists" in err
    assert csv_path.read_text() == original


def test_force_regenerates(tmp_path: Path) -> None:
    """--force overwrites an existing CSV based on the current raw directory."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "img_4x.png").write_text("")

    config = _make_config(tmp_path, raw_dir)
    assert main(["--config", str(config)]) == 0

    csv_path = raw_dir.parent / "metadata.csv"
    csv_path.write_text("stale,4x,\n")

    assert main(["--config", str(config), "--force"]) == 0
    assert "img_4x.png,4x," in csv_path.read_text()
    assert "stale" not in csv_path.read_text()


def test_update_appends_new_files_and_preserves_condition(tmp_path: Path) -> None:
    """--update adds new images without touching manually filled conditions."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "first_4x.png").write_text("")

    config = _make_config(tmp_path, raw_dir)
    assert main(["--config", str(config)]) == 0

    csv_path = raw_dir.parent / "metadata.csv"
    lines = _read_csv(csv_path)
    lines[1] = "first_4x.png,4x,ctrl"
    csv_path.write_text("\n".join(lines) + "\n")

    (raw_dir / "second_10x.png").write_text("")
    assert main(["--config", str(config), "--update"]) == 0

    text = csv_path.read_text()
    assert "first_4x.png,4x,ctrl" in text
    assert "second_10x.png,10x," in text


def test_update_warns_about_stale_rows(tmp_path: Path, capsys) -> None:
    """--update warns when the CSV references images no longer in raw/."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "present_4x.png").write_text("")

    config = _make_config(tmp_path, raw_dir)
    assert main(["--config", str(config)]) == 0

    csv_path = raw_dir.parent / "metadata.csv"
    csv_path.write_text(csv_path.read_text() + "missing_10x.png,10x,old\n")

    assert main(["--config", str(config), "--update"]) == 0
    out, err = capsys.readouterr()
    assert "missing_10x.png" in err
    assert "no longer in raw" in err.lower() or "stale" in err.lower()
    assert "missing_10x.png,10x,old" in csv_path.read_text()


def test_force_and_update_are_mutually_exclusive(tmp_path: Path) -> None:
    """Passing both --force and --update is rejected at the CLI level."""
    with pytest.raises(SystemExit):
        main(["--config", str(tmp_path / "config.yaml"), "--force", "--update"])


def test_raw_dir_override(tmp_path: Path) -> None:
    """--raw-dir overrides the raw directory from the config."""
    config_dir = tmp_path / "from_config"
    config_dir.mkdir()
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    (override_dir / "override_10x.png").write_text("")

    config = _make_config(tmp_path, config_dir)
    assert main(["--config", str(config), "--raw-dir", str(override_dir)]) == 0

    csv_path = override_dir.parent / "metadata.csv"
    assert "override_10x.png,10x," in csv_path.read_text()


def test_summary_includes_counts(tmp_path: Path, capsys) -> None:
    """The printed summary reports totals and per-magnification counts."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "a_4x.png").write_text("")
    (raw_dir / "b_10x.png").write_text("")
    (raw_dir / "no_suffix.png").write_text("")

    config = _make_config(tmp_path, raw_dir)
    assert main(["--config", str(config)]) == 0

    out, err = capsys.readouterr()
    assert "Total files" in out
    assert "4x" in out
    assert "10x" in out
