"""Annotation quality-control: validate raw/mask pairs and report class distributions."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tifffile

from spheroid_seg.data.dataset import IMAGE_EXTENSIONS

VALID_MASK_VALUES = {0, 1, 2, 3}


def _read_shape(path: Path) -> tuple[int, ...]:
    """Read only the shape of an image file without loading pixel data."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tiff:
            page = tiff.pages[0]
            return page.shape
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image.shape


def _read_mask(path: Path) -> np.ndarray:
    """Read a mask in its native dtype; raise if it cannot be interpreted as 2D."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        mask = tifffile.imread(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {path}")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape} for {path}")
    return mask


def _collect_pairs(raw_dir: Path, masks_dir: Path) -> list[tuple[Path, Path, str]]:
    """Collect raw/mask pairs by base name, ignoring unmatched files."""
    raw_by_stem: dict[str, Path] = {}
    for path in sorted(raw_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            raw_by_stem[path.stem] = path

    pairs: list[tuple[Path, Path, str]] = []
    for path in sorted(masks_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem in raw_by_stem:
            pairs.append((raw_by_stem[path.stem], path, path.stem))
    return pairs


def _validate_pair(
    raw_path: Path,
    mask_path: Path,
) -> list[str]:
    """Validate a single raw/mask pair; return a list of violation messages."""
    violations: list[str] = []

    try:
        raw_shape = _read_shape(raw_path)
    except Exception as exc:  # noqa: BLE001
        return [f"raw read error: {exc}"]

    try:
        mask = _read_mask(mask_path)
    except Exception as exc:  # noqa: BLE001
        return [f"mask read error: {exc}"]

    raw_hw = raw_shape[:2]
    mask_hw = mask.shape[:2]
    if raw_hw != mask_hw:
        violations.append(f"shape mismatch: raw {raw_hw} vs mask {mask_hw}")

    if mask.dtype != np.uint8:
        violations.append(f"mask dtype {mask.dtype} != uint8")

    unique_values = set(np.unique(mask).tolist())
    invalid_values = unique_values - VALID_MASK_VALUES
    if invalid_values:
        violations.append(f"invalid mask values {sorted(invalid_values)}")

    return violations


def _class_counts(mask: np.ndarray) -> dict[int, int]:
    """Count pixels for each valid class ID in a mask."""
    counts: dict[int, int] = dict.fromkeys(sorted(VALID_MASK_VALUES), 0)
    for value, count in zip(*np.unique(mask, return_counts=True), strict=False):
        if value in counts:
            counts[int(value)] = int(count)
    return counts


def run_qc(
    raw_dir: Path | str,
    masks_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    verbose: bool = True,
) -> Literal[0, 1]:
    """Run QC on all raw/mask pairs and optionally write a CSV report.

    Args:
        raw_dir: Directory containing raw images.
        masks_dir: Directory containing annotation masks.
        output_dir: Directory where the QC report CSV will be written.
        verbose: Whether to print per-image and global summaries.

    Returns:
        0 if no violations were found, 1 otherwise.
    """
    raw_dir = Path(raw_dir)
    masks_dir = Path(masks_dir)

    if not raw_dir.exists() or not masks_dir.exists():
        print("Error: raw and/or mask directory does not exist.", file=sys.stderr)
        return 1

    pairs = _collect_pairs(raw_dir, masks_dir)

    if not pairs:
        print(
            f"No paired raw/mask images found in '{raw_dir}' / '{masks_dir}'. "
            "Skipping QC (expected when real data is not yet present)."
        )
        return 0

    rows: list[dict[str, object]] = []
    global_counts: Counter[int] = Counter()
    total_violations = 0

    for raw_path, mask_path, name in pairs:
        violations = _validate_pair(raw_path, mask_path)
        mask = _read_mask(mask_path)
        counts = _class_counts(mask)
        global_counts.update(counts)

        row: dict[str, object] = {
            "image": name,
            "total_pixels": sum(counts.values()),
            **{f"class_{cls}": counts[cls] for cls in sorted(VALID_MASK_VALUES)},
            "violations": "; ".join(violations) if violations else "",
        }
        rows.append(row)

        if verbose:
            total = sum(counts.values())
            dist = ", ".join(
                f"{cls}: {counts[cls]} ({100 * counts[cls] / total:.2f}%)"
                for cls in sorted(VALID_MASK_VALUES)
            )
            status = "OK" if not violations else "FAIL"
            print(f"[{status}] {name}: {dist}")
            for violation in violations:
                print(f"       -> {violation}")

        if violations:
            total_violations += 1

    if verbose and global_counts:
        global_total = sum(global_counts.values())
        print("\nGlobal class distribution:")
        for cls in sorted(VALID_MASK_VALUES):
            count = global_counts[cls]
            pct = 100 * count / global_total if global_total else 0.0
            print(f"  class {cls}: {count} ({pct:.2f}%)")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "qc_report.csv"
        with report_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "total_pixels",
                    "class_0",
                    "class_1",
                    "class_2",
                    "class_3",
                    "violations",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        if verbose:
            print(f"\nQC report written to {report_path}")

    if total_violations:
        print(
            f"\nQC failed: {total_violations}/{len(pairs)} image(s) violated the annotation spec.",
            file=sys.stderr,
        )
        return 1

    print(f"\nQC passed: {len(pairs)} image(s) validated.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the QC tool."""
    parser = argparse.ArgumentParser(
        description="Validate raw/mask annotation pairs and report class distributions."
    )
    parser.add_argument("--raw-dir", required=True, help="Directory with raw images.")
    parser.add_argument("--mask-dir", required=True, help="Directory with annotation masks.")
    parser.add_argument(
        "--output-dir",
        default="outputs/qc",
        help="Directory for the QC report CSV (default: outputs/qc).",
    )
    args = parser.parse_args(argv)

    return run_qc(args.raw_dir, args.mask_dir, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
