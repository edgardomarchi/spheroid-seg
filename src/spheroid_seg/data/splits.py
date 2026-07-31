"""Train/validation/test split file utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

SPLIT_NAMES = ("train", "val", "test")


def load_split_list(splits_dir: Path | str, split_name: str) -> list[str]:
    """Read a split file containing one base image name per line.

    Args:
        splits_dir: Directory containing ``{split_name}.txt`` files.
        split_name: Name of the split (e.g. ``train``).

    Returns:
        List of base image names (stems) in the split.
    """
    path = Path(splits_dir) / f"{split_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    with path.open("r") as f:
        names = [line.strip() for line in f if line.strip()]

    # Preserve order while removing accidental duplicates within the same file.
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    return unique_names


def check_split_leak(splits: dict[str, list[str]]) -> None:
    """Raise a clear error if the same image appears in more than one split.

    Args:
        splits: Mapping from split name to list of base image names.

    Raises:
        ValueError: If any base image name is present in multiple splits.
    """
    membership: dict[str, str] = {}
    for split_name, names in splits.items():
        for name in names:
            if name in membership:
                raise ValueError(
                    f"Patch-level leak detected: image '{name}' appears in both "
                    f"'{membership[name]}' and '{split_name}' splits. "
                    "Data splits must be image-level and mutually exclusive."
                )
            membership[name] = split_name


def load_splits(
    splits_dir: Path | str,
    split_names: Iterable[str] = SPLIT_NAMES,
) -> dict[str, list[str]]:
    """Load all requested split files and verify there are no image-level leaks.

    Args:
        splits_dir: Directory containing ``{split_name}.txt`` files.
        split_names: Iterable of split names to load.

    Returns:
        Mapping from split name to ordered list of base image names.
    """
    splits_dir = Path(splits_dir)
    splits = {name: load_split_list(splits_dir, name) for name in split_names}
    check_split_leak(splits)
    return splits
