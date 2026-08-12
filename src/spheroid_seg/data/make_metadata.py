"""Generate data/metadata.csv from raw-image filenames.

This module is used by ``scripts/make_metadata.py``; core logic lives here so it
is importable and unit-testable without relying on subprocess path tricks.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from spheroid_seg.data.dataset import IMAGE_EXTENSIONS
from spheroid_seg.data.metadata import parse_magnification


def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML configuration file."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def _collect_raw_images(raw_dir: Path) -> list[Path]:
    """Return image files in *raw_dir*, sorted alphabetically by filename."""
    if not raw_dir.exists():
        return []
    files = [
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name)


def _parse_filename_magnification(filename: str) -> str:
    """Return the magnification suffix (``4x`` or ``10x``) or an empty string."""
    mag = parse_magnification(filename)
    return "" if mag == "unknown" else mag


def _build_rows(raw_dir: Path) -> list[dict[str, str]]:
    """Build metadata rows from the files in *raw_dir*."""
    rows: list[dict[str, str]] = []
    for path in _collect_raw_images(raw_dir):
        rows.append(
            {
                "image": path.name,
                "magnification": _parse_filename_magnification(path.name),
                "condition": "",
            }
        )
    return sorted(rows, key=lambda row: row["image"])


def _read_existing_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    """Load existing metadata rows keyed by image filename.

    Returns:
        Mapping from image filename to row dict. Missing or unsupported
        magnification values are preserved as-is so manual edits are not lost.
    """
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = (row.get("image") or "").strip()
            if not image:
                continue
            rows[image] = {
                "image": image,
                "magnification": (row.get("magnification") or "").strip(),
                "condition": (row.get("condition") or "").strip(),
            }
    return rows


def _write_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows to *csv_path* with the standard header."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "magnification", "condition"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _warn_unrecognized(rows: list[dict[str, str]], file: Any) -> list[dict[str, str]]:
    """Print a warning for rows with an empty magnification and return them."""
    unrecognized = [row for row in rows if not row["magnification"]]
    if unrecognized:
        names = ", ".join(row["image"] for row in unrecognized)
        print(
            f"WARNING: {len(unrecognized)} file(s) need manual magnification: {names}",
            file=file,
        )
    return unrecognized


def _magnification_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    """Return counts per magnification, treating empty values as 'unspecified'."""
    counts: Counter[str] = Counter()
    for row in rows:
        mag = row["magnification"].strip() or "unspecified"
        counts[mag] += 1
    return dict(sorted(counts.items()))


def _print_summary(
    total_files: int,
    rows: list[dict[str, str]],
    added: int,
    update: bool,
) -> None:
    """Print a human-readable summary of the metadata CSV."""
    action = "added" if update else "written"
    print("Metadata summary:")
    print(f"  Total files scanned: {total_files}")
    print(f"  Rows {action}: {len(rows)} ({added} new)")
    for mag, count in _magnification_counts(rows).items():
        print(f"  {mag}: {count}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate data/metadata.csv from raw-image filenames."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="YAML config file (default: configs/base.yaml).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override the raw image directory from the config.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing metadata CSV.",
    )
    group.add_argument(
        "--update",
        action="store_true",
        help=(
            "Merge new raw files into an existing metadata CSV, preserving "
            "existing rows and their conditions."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for generating data/metadata.csv.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 on success, non-zero on error).
    """
    args = parse_args(argv)

    try:
        raw_dir = args.raw_dir
        if raw_dir is None:
            config = load_config(args.config)
            raw_dir = Path(config["data"]["raw_dir"])

        csv_path = raw_dir.parent / "metadata.csv"
        new_rows = _build_rows(raw_dir)

        existing: dict[str, dict[str, str]] = {}
        stale: list[str] = []
        added = len(new_rows)

        if csv_path.exists():
            if args.force:
                pass  # Overwrite below.
            elif args.update:
                existing = _read_existing_rows(csv_path)
                current_images = {row["image"] for row in new_rows}
                stale = sorted(image for image in existing if image not in current_images)
                if stale:
                    print(
                        "WARNING: stale rows in metadata CSV (image no longer in raw/): "
                        + ", ".join(stale),
                        file=sys.stderr,
                    )
                merged = {**existing}
                for row in new_rows:
                    if row["image"] not in merged:
                        merged[row["image"]] = row
                new_rows = sorted(merged.values(), key=lambda r: r["image"])
                added = sum(1 for row in new_rows if row["image"] not in existing)
            else:
                print(
                    f"metadata CSV already exists: {csv_path}. "
                    "Pass --force to overwrite or --update to merge.",
                    file=sys.stderr,
                )
                return 1

        existing_keys = set(existing.keys())
        _warn_unrecognized(
            [row for row in new_rows if row["image"] not in existing_keys], sys.stderr
        )

        _write_csv(csv_path, new_rows)
        _print_summary(len(_collect_raw_images(raw_dir)), new_rows, added, args.update)

    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"Invalid config: missing key {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
