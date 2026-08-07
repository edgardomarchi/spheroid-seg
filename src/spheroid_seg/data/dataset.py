"""Paired raw/mask image loading and preprocessing utilities."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tifffile

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def has_real_pairs(raw_dir: Path | str, masks_dir: Path | str) -> bool:
    """Return True if at least one raw/mask pair with matching stem exists.

    Split files alone do not constitute real data; this checks for actual
    image files in ``raw_dir`` and ``masks_dir`` that share the same base name.
    """
    raw_dir = Path(raw_dir)
    masks_dir = Path(masks_dir)
    if not raw_dir.exists() or not masks_dir.exists():
        return False

    raw_by_stem: dict[str, Path] = {}
    for path in raw_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            raw_by_stem[path.stem] = path

    if not raw_by_stem:
        return False

    return any(
        path.stem in raw_by_stem
        for path in masks_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_image(path: Path) -> np.ndarray:
    """Read an image file, returning an HxW grayscale or HxW[xC] array."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image = tifffile.imread(path)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.ndim not in {2, 3}:
        raise ValueError(f"Unsupported image shape {image.shape} for {path}")
    return image


def _read_mask(path: Path) -> np.ndarray:
    """Read a single-channel mask file as uint8."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        mask = tifffile.imread(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {path}")

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape} for {path}")
    return mask.astype(np.uint8)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to a single-channel grayscale array."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Cannot convert image with shape {image.shape} to grayscale")


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image to a 3-channel RGB array."""
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    gray = _to_grayscale(image)
    return np.stack([gray, gray, gray], axis=-1)


def normalize_percentile(image: np.ndarray, low: int = 1, high: int = 99) -> np.ndarray:
    """Normalize image intensities to [0, 1] using percentiles.

    Args:
        image: Input image, any numeric dtype.
        low: Lower percentile for clipping.
        high: Upper percentile for clipping.

    Returns:
        Float32 array in [0, 1] with same channel layout as input.
    """
    image = image.astype(np.float32)
    p_low, p_high = np.percentile(image, [low, high])
    if p_high <= p_low:
        return np.zeros_like(image, dtype=np.float32)
    clipped = np.clip(image, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low)).astype(np.float32)


def remap_classes(mask: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Remap mask pixel values using a lookup table.

    Args:
        mask: Uint8 mask array.
        mapping: Dictionary mapping old class IDs to new class IDs.

    Returns:
        Remapped uint8 mask.
    """
    lut = np.arange(256, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[src] = dst
    return lut[mask]


def load_pair(
    raw_path: Path,
    mask_path: Path,
    *,
    input_channels: Literal["rgb", "grayscale"],
    class_mapping: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess a single raw/mask pair.

    Args:
        raw_path: Path to the raw image.
        mask_path: Path to the corresponding mask.
        input_channels: Whether to return "rgb" or "grayscale" images.
        class_mapping: Optional class remapping; defaults to identity.

    Returns:
        Tuple of (image, mask) where image is float32 in [0, 1] and mask is uint8.
    """
    image = _read_image(raw_path)
    image = normalize_percentile(image)

    if input_channels == "grayscale":
        image = _to_grayscale(image)
    elif input_channels == "rgb":
        image = _to_rgb(image)
    else:
        raise ValueError(f"input_channels must be 'rgb' or 'grayscale', got {input_channels}")

    mask = _read_mask(mask_path)
    if class_mapping is not None:
        mask = remap_classes(mask, class_mapping)

    return image.astype(np.float32), mask


class SpheroidDataset:
    """Dataset of paired raw images and segmentation masks.

    Attributes:
        pairs: List of (raw_path, mask_path, base_name) tuples.
        input_channels: "rgb" or "grayscale".
        class_mapping: Class remapping dictionary.
    """

    def __init__(
        self,
        raw_dir: Path | str,
        masks_dir: Path | str,
        *,
        input_channels: Literal["rgb", "grayscale"] = "grayscale",
        class_mapping: dict[int, int] | None = None,
    ) -> None:
        """Initialize the dataset by pairing raw images with masks by base name."""
        self.raw_dir = Path(raw_dir)
        self.masks_dir = Path(masks_dir)
        self.input_channels = input_channels
        self.class_mapping = class_mapping
        self.pairs = self._collect_pairs()

    def _collect_pairs(self) -> list[tuple[Path, Path, str]]:
        """Collect all raw/mask pairs sharing the same base name."""
        raw_by_stem: dict[str, Path] = {}
        for path in sorted(self.raw_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                raw_by_stem[path.stem] = path

        pairs: list[tuple[Path, Path, str]] = []
        for path in sorted(self.masks_dir.iterdir()):
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and path.stem in raw_by_stem
            ):
                pairs.append((raw_by_stem[path.stem], path, path.stem))

        return pairs

    def __len__(self) -> int:
        """Number of paired samples."""
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        """Load the image and mask at the given index.

        Returns:
            Tuple of (image, mask, base_name).
        """
        raw_path, mask_path, name = self.pairs[index]
        image, mask = load_pair(
            raw_path,
            mask_path,
            input_channels=self.input_channels,
            class_mapping=self.class_mapping,
        )
        return image, mask, name

    def iter_pairs(self) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
        """Yield all loaded pairs."""
        for idx in range(len(self)):
            yield self[idx]
