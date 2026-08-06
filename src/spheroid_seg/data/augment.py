"""Albumentations-based augmentation pipeline for image/mask pairs."""

from __future__ import annotations

from typing import Any

import albumentations as A  # noqa: N812
import cv2
import numpy as np


def build_augmentation(config: dict[str, Any], seed: int | None = None) -> A.Compose:
    """Build a deterministic augmentation pipeline from a config dictionary.

    The pipeline applies spatial and photometric augmentations jointly to image
    and mask pairs. Output patches keep the input ``patch_size``.

    Args:
        config: Augmentation configuration (typically ``config["augment"]``).
        seed: Optional seed passed to Albumentations for reproducibility.

    Returns:
        An Albumentations ``Compose`` transform.
    """
    scale_range = tuple(config.get("scale_range", [0.5, 2.0]))
    if len(scale_range) != 2:
        raise ValueError("scale_range must be a [min, max] tuple")

    transforms = [
        A.HorizontalFlip(p=config.get("flip_probability", 0.5)),
        A.VerticalFlip(p=config.get("flip_probability", 0.5)),
        A.RandomRotate90(p=config.get("rotate90_probability", 0.5)),
        A.ElasticTransform(
            alpha=config.get("elastic_alpha", 1),
            sigma=config.get("elastic_sigma", 50),
            p=config.get("elastic_probability", 0.3),
        ),
        A.Affine(
            scale=scale_range,
            keep_ratio=True,
            p=config.get("scale_probability", 0.5),
            border_mode=cv2.BORDER_REFLECT_101,
            fill=0,
        ),
        A.GaussianBlur(
            blur_limit=tuple(config.get("blur_limit", [3, 7])),
            sigma_limit=tuple(config.get("blur_sigma_limit", [0.1, 2.0])),
            p=config.get("blur_probability", 0.2),
        ),
        A.GaussNoise(
            std_range=tuple(config.get("noise_std_range", [0.01, 0.05])),
            p=config.get("noise_probability", 0.2),
        ),
        A.RandomBrightnessContrast(
            brightness_limit=config.get("brightness_limit", 0.3),
            contrast_limit=config.get("contrast_limit", 0.3),
            p=config.get("brightness_contrast_probability", 0.5),
        ),
    ]

    return A.Compose(
        transforms,
        seed=seed,
        strict=False,
    )


def apply_augmentation(
    transform: A.Compose,
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an Albumentations transform to an image/mask pair.

    Args:
        transform: Albumentations ``Compose`` pipeline.
        image: HxW[xC] float or uint8 image.
        mask: HxW uint8 mask.

    Returns:
        Augmented (image, mask) with the same shapes/dtypes as input.
    """
    result = transform(image=image, mask=mask)
    return result["image"], result["mask"]
