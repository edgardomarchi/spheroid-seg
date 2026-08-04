"""Create deterministic, stratified train/val/test split files for real data.

This module is used by ``scripts/make_splits.py``; core logic lives here so it is
importable and unit-testable without relying on subprocess path tricks.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from spheroid_seg.data.dataset import IMAGE_EXTENSIONS
from spheroid_seg.data.metadata import load_metadata_table
from spheroid_seg.data.splits import load_splits

SPLIT_NAMES = ("train", "val", "test")


def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML configuration file."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def _collect_files_by_stem(directory: Path) -> dict[str, Path]:
    """Return a mapping from file stem to path for image files in *directory*."""
    files: dict[str, Path] = {}
    if directory.exists():
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                files[path.stem] = path
    return files


def validate_pairs(
    raw_dir: Path,
    masks_dir: Path,
    metadata: dict[str, dict[str, str | None]],
) -> None:
    """Validate that every metadata row has paired raw/mask files.

    Orphaned raw files, orphaned mask files, and metadata rows without files are
    reported together so the user gets a complete picture of the mismatch.

    Args:
        raw_dir: Directory containing raw images.
        masks_dir: Directory containing annotation masks.
        metadata: Mapping from image base name to ``{"magnification": ..., "condition": ...}``.

    Raises:
        ValueError: If any inconsistency is found between metadata, raw files, and masks.
    """
    raw_by_stem = _collect_files_by_stem(raw_dir)
    masks_by_stem = _collect_files_by_stem(masks_dir)

    metadata_without_files: list[str] = []
    for stem in sorted(metadata):
        has_raw = stem in raw_by_stem
        has_mask = stem in masks_by_stem
        if not has_raw or not has_mask:
            metadata_without_files.append(f"  {stem}: raw={has_raw}, mask={has_mask}")

    raw_without_mask = sorted(stem for stem in raw_by_stem if stem not in masks_by_stem)
    mask_without_raw = sorted(stem for stem in masks_by_stem if stem not in raw_by_stem)

    issues: list[str] = []
    if metadata_without_files:
        issues.append("Metadata rows missing raw/mask files:\n" + "\n".join(metadata_without_files))
    if raw_without_mask:
        issues.append(
            f"Raw files without matching masks ({len(raw_without_mask)}): "
            + ", ".join(raw_without_mask)
        )
    if mask_without_raw:
        issues.append(
            f"Mask files without matching raw files ({len(mask_without_raw)}): "
            + ", ".join(mask_without_raw)
        )

    if issues:
        raise ValueError(
            "Split creation aborted because raw/mask/metadata are inconsistent.\n"
            + "\n\n".join(issues)
        )


def _eligible_for_joint_stratification(
    metadata: dict[str, dict[str, str | None]],
    train_frac: float,
    val_frac: float,
) -> bool:
    """Return True if every (magnification, condition) group can fill all splits.

    With the default rounding rule, a group needs at least 6 samples for the
    70/15/15 split to place one image in each of train/val/test. The threshold
    is computed generically from the requested fractions so custom fractions
    behave consistently.
    """
    group_counts: dict[tuple[str, str], int] = defaultdict(int)
    for _stem, row in metadata.items():
        condition = row.get("condition") or "unknown"
        group_counts[(row["magnification"], condition)] += 1

    if not group_counts:
        return False

    min_required = 3
    for count in group_counts.values():
        if count < min_required:
            return False
        n_train = int(round(count * train_frac))
        n_val = int(round(count * val_frac))
        n_test = count - n_train - n_val
        if n_train < 1 or n_val < 1 or n_test < 1:
            return False
    return True


def determine_stratification(
    metadata: dict[str, dict[str, str | None]],
    train_frac: float,
    val_frac: float,
) -> tuple[list[tuple[str, ...]], bool]:
    """Choose stratification keys and whether the condition column is active.

    Returns:
        A tuple of (list of stratification group keys, use_condition flag).
        Each key is ``(magnification,)`` or ``(magnification, condition)``.
    """
    has_condition = any((row.get("condition") or "").strip() for row in metadata.values())
    if has_condition and _eligible_for_joint_stratification(metadata, train_frac, val_frac):
        keys = sorted(
            {
                (row["magnification"], (row.get("condition") or "unknown").strip())
                for row in metadata.values()
            }
        )
        return keys, True

    keys = sorted({(row["magnification"],) for row in metadata.values()})
    return keys, False


def _group_names_by_key(
    metadata: dict[str, dict[str, str | None]],
    keys: list[tuple[str, ...]],
) -> dict[tuple[str, ...], list[str]]:
    """Assign image base names to stratification groups."""
    groups: dict[tuple[str, ...], list[str]] = {key: [] for key in keys}
    for stem in sorted(metadata):
        row = metadata[stem]
        if len(keys[0]) == 2:
            key = (row["magnification"], (row.get("condition") or "unknown").strip())
        else:
            key = (row["magnification"],)
        groups[key].append(stem)
    return groups


def assign_splits(
    metadata: dict[str, dict[str, str | None]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[dict[str, list[str]], bool]:
    """Create stratified train/val/test assignments.

    The rounding rule per group is:

    * ``n_train = int(round(n * train_frac))``
    * ``n_val   = int(round(n * val_frac))``
    * ``n_test  = n - n_train - n_val``

    Counts therefore sum exactly to the group size. Groups are processed in
    sorted key order; names are sorted within each group before shuffling so the
    result is byte-identical for the same inputs and seed.

    Args:
        metadata: Mapping from image base name to row data.
        train_frac: Fraction of images for training.
        val_frac: Fraction of images for validation.
        seed: Random seed for deterministic shuffling.

    Returns:
        A tuple of (splits mapping, use_condition flag).
    """
    keys, use_condition = determine_stratification(metadata, train_frac, val_frac)
    groups = _group_names_by_key(metadata, keys)

    rng = np.random.default_rng(seed)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for key in sorted(groups):
        names = sorted(groups[key])
        rng.shuffle(names)
        n = len(names)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        splits["train"].extend(names[:n_train])
        splits["val"].extend(names[n_train : n_train + n_val])
        splits["test"].extend(names[n_train + n_val :])

    return splits, use_condition


def write_splits(
    splits: dict[str, list[str]],
    splits_dir: Path,
    force: bool,
) -> None:
    """Write the three split files, refusing to overwrite unless *force* is True.

    Args:
        splits: Mapping from split name to ordered list of base names.
        splits_dir: Directory for ``{train,val,test}.txt``.
        force: If True, overwrite existing files.

    Raises:
        FileExistsError: If any split file already exists and *force* is False.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    existing = [name for name in SPLIT_NAMES if (splits_dir / f"{name}.txt").exists()]
    if existing and not force:
        raise FileExistsError(
            "Split files already exist and would be overwritten: "
            + ", ".join(f"{name}.txt" for name in existing)
            + ". Pass --force to overwrite."
        )

    for name in SPLIT_NAMES:
        path = splits_dir / f"{name}.txt"
        path.write_text("\n".join(splits[name]) + ("\n" if splits[name] else ""))


def _magnification_counts(
    names: list[str],
    metadata: dict[str, dict[str, str | None]],
) -> dict[str, int]:
    """Return counts per magnification for the given names."""
    counts: dict[str, int] = defaultdict(int)
    for name in names:
        counts[metadata[name]["magnification"]] += 1
    return dict(sorted(counts.items()))


def _condition_counts(
    names: list[str],
    metadata: dict[str, dict[str, str | None]],
) -> dict[str, int]:
    """Return counts per condition for the given names."""
    counts: dict[str, int] = defaultdict(int)
    for name in names:
        condition = (metadata[name].get("condition") or "unknown").strip()
        counts[condition] += 1
    return dict(sorted(counts.items()))


def print_summary(
    splits: dict[str, list[str]],
    metadata: dict[str, dict[str, str | None]],
    use_condition: bool,
) -> None:
    """Print a human-readable summary of the created splits."""
    print("Split summary:")
    for name in SPLIT_NAMES:
        names = splits[name]
        print(f"  {name}: {len(names)} images")
        mag_counts = _magnification_counts(names, metadata)
        for mag, count in mag_counts.items():
            print(f"    {mag}: {count}")
        if use_condition:
            cond_counts = _condition_counts(names, metadata)
            for condition, count in cond_counts.items():
                print(f"    condition={condition}: {count}")


def _validate_fractions(train_frac: float, val_frac: float) -> None:
    """Ensure split fractions are in (0, 1) and leave room for a test set."""
    for name, value in (("train-frac", train_frac), ("val-frac", val_frac)):
        if not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1), got {value}")
    if train_frac + val_frac >= 1:
        raise ValueError(
            f"train-frac ({train_frac}) + val-frac ({val_frac}) must sum to less than 1"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test split files from real data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config file (must contain seed and data paths).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/metadata.csv"),
        help=(
            "Metadata CSV with columns image,magnification[,condition] "
            "(default: data/metadata.csv)."
        ),
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Fraction of images for training (default: 0.70).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of images for validation (default: 0.15).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for creating stratified split files.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 on success, non-zero on error).
    """
    args = parse_args(argv)

    try:
        _validate_fractions(args.train_frac, args.val_frac)
        config = load_config(args.config)
        seed = config["seed"]

        raw_dir = Path(config["data"]["raw_dir"])
        masks_dir = Path(config["data"]["masks_dir"])
        splits_dir = Path(config["data"]["splits_dir"])

        metadata = load_metadata_table(args.metadata)
        if not metadata:
            print(f"No rows found in metadata CSV: {args.metadata}", file=sys.stderr)
            return 1

        validate_pairs(raw_dir, masks_dir, metadata)

        splits, use_condition = assign_splits(metadata, args.train_frac, args.val_frac, seed)
        if use_condition:
            print("Stratifying jointly by magnification and condition.")
        else:
            has_condition = any((row.get("condition") or "").strip() for row in metadata.values())
            if has_condition:
                print(
                    "WARNING: condition column present but at least one "
                    "(magnification, condition) group is too small for joint "
                    "stratification; falling back to magnification-only stratification.",
                    file=sys.stderr,
                )
            else:
                print("Stratifying by magnification.")

        write_splits(splits, splits_dir, args.force)

        # Verify with the existing loader/leak detector.
        loaded = load_splits(splits_dir)
        if {len(loaded[name]) for name in SPLIT_NAMES} != {
            len(splits[name]) for name in SPLIT_NAMES
        }:
            raise RuntimeError("Split verification failed after writing.")

        print_summary(splits, metadata, use_condition)

    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
