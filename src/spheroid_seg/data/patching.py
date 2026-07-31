"""Patch extraction with oversampling of object-containing regions."""

from __future__ import annotations

import numpy as np


def _pad_if_needed(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reflect-pad image and mask so both dimensions are at least patch_size."""
    h, w = image.shape[:2]
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h == 0 and pad_w == 0:
        return image, mask

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        if image.ndim == 3
        else ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="reflect",
    )
    mask = np.pad(
        mask,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="reflect",
    )
    return image, mask


def _is_object_patch(mask_patch: np.ndarray, min_object_fraction: float) -> bool:
    """Return True if the fraction of foreground pixels exceeds the threshold."""
    foreground = (mask_patch > 0).sum()
    return foreground / mask_patch.size > min_object_fraction


def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    patch_size: int,
    min_object_fraction: float,
    object_patch_ratio: float,
    patches_per_image: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract patches from an image/mask pair with object oversampling.

    Patches are sampled uniformly at random from the image. The final set is
    composed to match ``object_patch_ratio`` as closely as possible given the
    available object-containing patches. Sampling is deterministic given
    ``seed``.

    Args:
        image: HxW[xC] image array.
        mask: HxW mask array.
        patch_size: Side length of square patches.
        min_object_fraction: Minimum foreground fraction for a patch to count as
            object-containing.
        object_patch_ratio: Target fraction of object patches in the returned set.
        patches_per_image: Number of patches to extract from this image.
        seed: Random seed for reproducible sampling.

    Returns:
        Tuple of (image_patches, mask_patches) with shapes
        (patches_per_image, patch_size, patch_size, C) and
        (patches_per_image, patch_size, patch_size).
    """
    rng = np.random.default_rng(seed)
    image, mask = _pad_if_needed(image, mask, patch_size)
    h, w = image.shape[:2]

    target_object = int(round(patches_per_image * object_patch_ratio))
    target_background = patches_per_image - target_object

    object_patches: list[tuple[np.ndarray, np.ndarray]] = []
    background_patches: list[tuple[np.ndarray, np.ndarray]] = []

    max_attempts = max(patches_per_image * 10, 1000)
    for _ in range(max_attempts):
        if (
            len(object_patches) >= target_object
            and len(background_patches) >= target_background
        ):
            break

        y = rng.integers(0, h - patch_size + 1)
        x = rng.integers(0, w - patch_size + 1)
        patch_image = image[y : y + patch_size, x : x + patch_size]
        patch_mask = mask[y : y + patch_size, x : x + patch_size]

        if _is_object_patch(patch_mask, min_object_fraction):
            if len(object_patches) < target_object:
                object_patches.append((patch_image, patch_mask))
        else:
            if len(background_patches) < target_background:
                background_patches.append((patch_image, patch_mask))

    # If one pool is too small, fill from the other to reach the requested count.
    while len(object_patches) < target_object and background_patches:
        object_patches.append(background_patches.pop())
    while len(background_patches) < target_background and object_patches:
        background_patches.append(object_patches.pop())

    selected = object_patches + background_patches
    rng.shuffle(selected)

    if len(selected) < patches_per_image:
        # Last resort: duplicate existing patches to honor the count.
        extra = rng.choice(len(selected), size=patches_per_image - len(selected))
        selected.extend(selected[i] for i in extra)

    selected = selected[:patches_per_image]
    image_patches = np.stack([p[0] for p in selected])
    mask_patches = np.stack([p[1] for p in selected])
    return image_patches, mask_patches
