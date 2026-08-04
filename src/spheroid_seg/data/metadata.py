"""Magnification metadata parsing and optional CSV loader."""

from __future__ import annotations

from pathlib import Path

# Suffix is the token immediately before the file extension. Recognized values
# are exactly "4x" and "10x" when preceded by an underscore.
_MAGNIFICATION_SUFFIXES = {"4x", "10x"}


def parse_magnification(image_name: str) -> str:
    """Return the magnification group from a filename suffix.

    The suffix is the token immediately before any file extension. Only the
    exact suffixes ``_4x`` and ``_10x`` are recognized (case-sensitive).

    Args:
        image_name: Filename or base name (e.g. ``"synth_000_4x.png"``).

    Returns:
        ``"4x"``, ``"10x"``, or ``"unknown"``.
    """
    stem = Path(image_name).stem
    for suffix in _MAGNIFICATION_SUFFIXES:
        if stem.endswith(f"_{suffix}"):
            return suffix
    return "unknown"


def _resolve_name_column(header: list[str]) -> int:
    """Return the index of the image-name column, accepting ``image`` or ``image_name``."""
    if "image" in header:
        return header.index("image")
    if "image_name" in header:
        return header.index("image_name")
    raise ValueError(
        f"Metadata CSV must contain 'image' (or 'image_name') and 'magnification' columns, "
        f"got: {header}"
    )


def _parse_metadata_rows(
    f,
    name_idx: int,
    mag_idx: int,
    condition_idx: int | None,
) -> dict[str, dict[str, str | None]]:
    """Parse CSV body rows into a mapping keyed by image base name."""
    metadata: dict[str, dict[str, str | None]] = {}
    for line_no, line in enumerate(f, start=2):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            raise ValueError(f"Malformed row {line_no} in metadata CSV: {line}")
        image_name = parts[name_idx].strip()
        magnification = parts[mag_idx].strip()
        if not image_name:
            raise ValueError(f"empty image_name at row {line_no} in metadata CSV")
        if magnification not in _MAGNIFICATION_SUFFIXES:
            raise ValueError(
                f"Invalid magnification '{magnification}' for image '{image_name}' "
                f"at row {line_no}; must be one of {_MAGNIFICATION_SUFFIXES}"
            )
        stem = Path(image_name).stem
        condition = parts[condition_idx].strip() if condition_idx is not None else None
        metadata[stem] = {"magnification": magnification, "condition": condition}
    return metadata


def load_metadata_table(csv_path: Path | str) -> dict[str, dict[str, str | None]]:
    """Load a metadata CSV with ``image,magnification`` and optional ``condition`` columns.

    The ``image`` column may also be spelled ``image_name`` for backward
    compatibility. Image values are normalised to their file stem so they match
    raw/mask base names regardless of extension.

    Args:
        csv_path: Path to the metadata CSV file.

    Returns:
        Mapping from image base name to ``{"magnification": str, "condition": str | None}``.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the header is missing required columns, a row has an
            empty image name, or a magnification value is not supported.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    with csv_path.open("r") as f:
        header = [h.strip() for h in f.readline().split(",")]
        if "magnification" not in header:
            raise ValueError(
                f"Metadata CSV must contain 'image' (or 'image_name') and 'magnification' columns, "
                f"got: {header}"
            )
        name_idx = _resolve_name_column(header)
        mag_idx = header.index("magnification")
        condition_idx = header.index("condition") if "condition" in header else None
        return _parse_metadata_rows(f, name_idx, mag_idx, condition_idx)


def load_metadata_csv(csv_path: Path | str) -> dict[str, str]:
    """Load an optional magnification CSV with ``image,magnification`` columns.

    The ``image`` column may also be spelled ``image_name`` for backward
    compatibility. The returned mapping overrides filename-derived magnification.
    Unknown magnification values or malformed rows raise a clear error.

    Args:
        csv_path: Path to the metadata CSV file.

    Returns:
        Mapping from image base name to magnification group.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the header is missing required columns, a row has an
            empty image name, or a magnification value is not supported.
    """
    table = load_metadata_table(csv_path)
    return {stem: row["magnification"] for stem, row in table.items()}


def build_magnification_map(
    image_names: list[str],
    csv_path: Path | str | None = None,
) -> dict[str, str]:
    """Resolve magnification for a list of images using CSV > filename > unknown.

    Args:
        image_names: List of image filenames or base names.
        csv_path: Optional metadata CSV path.

    Returns:
        Mapping from each image name to its magnification group.
    """
    csv_metadata = load_metadata_csv(csv_path) if csv_path is not None else {}
    return {
        name: csv_metadata.get(Path(name).stem, parse_magnification(name)) for name in image_names
    }
