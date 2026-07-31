"""Segmentation losses: soft Dice + class-weighted cross-entropy."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def dice_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """Multiclass soft Dice loss.

    Args:
        logits: Model logits of shape (..., num_classes).
        targets: Integer class map of shape (...).
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar Dice loss.
    """
    num_classes = logits.shape[-1]
    probs = jax.nn.softmax(logits, axis=-1)
    targets_one_hot = jax.nn.one_hot(targets, num_classes=num_classes)

    intersection = jnp.sum(probs * targets_one_hot, axis=tuple(range(logits.ndim - 1)))
    denominator = jnp.sum(
        jnp.square(probs) + jnp.square(targets_one_hot),
        axis=tuple(range(logits.ndim - 1)),
    )

    dice_per_class = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - jnp.mean(dice_per_class)


def cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    class_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Class-weighted cross-entropy loss.

    Args:
        logits: Model logits of shape (..., num_classes).
        targets: Integer class map of shape (...).
        class_weights: Optional array of shape (num_classes,) giving per-class
            loss weights. If None, all classes are weighted equally.

    Returns:
        Scalar cross-entropy loss.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    num_classes = logits.shape[-1]
    targets_one_hot = jax.nn.one_hot(targets, num_classes=num_classes)
    pixel_loss = -jnp.sum(targets_one_hot * log_probs, axis=-1)

    if class_weights is not None:
        weights = jnp.asarray(class_weights)[targets]
        pixel_loss = pixel_loss * weights

    return jnp.mean(pixel_loss)


def segmentation_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    class_weights: jnp.ndarray | None = None,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """Combined Dice + class-weighted cross-entropy loss.

    Args:
        logits: Model logits of shape (..., num_classes).
        targets: Integer class map of shape (...).
        class_weights: Optional per-class CE weights.
        dice_weight: Scalar weight for the Dice term.
        ce_weight: Scalar weight for the CE term.
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar combined loss.
    """
    return dice_weight * dice_loss(
        logits, targets, epsilon=epsilon
    ) + ce_weight * cross_entropy_loss(logits, targets, class_weights=class_weights)
