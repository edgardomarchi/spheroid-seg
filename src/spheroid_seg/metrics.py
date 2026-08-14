"""Per-class segmentation metrics: Dice and IoU."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _per_class_counts(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    num_classes: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return per-class intersections, predictions, and targets.

    Args:
        predictions: Integer class predictions of shape (...).
        targets: Integer class targets of shape (...).
        num_classes: Number of classes.

    Returns:
        Tuple of (intersection, prediction_area, target_area), each of shape
        (num_classes,).
    """
    pred_one_hot = jax.nn.one_hot(predictions, num_classes=num_classes, dtype=jnp.uint32)
    target_one_hot = jax.nn.one_hot(targets, num_classes=num_classes, dtype=jnp.uint32)

    axes = tuple(range(predictions.ndim))
    intersection = jnp.sum(pred_one_hot * target_one_hot, axis=axes, dtype=jnp.uint32)
    prediction_area = jnp.sum(pred_one_hot, axis=axes, dtype=jnp.uint32)
    target_area = jnp.sum(target_one_hot, axis=axes, dtype=jnp.uint32)

    return intersection, prediction_area, target_area


def dice_score(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    num_classes: int,
    *,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """Per-class Dice score.

    Empty-class behavior:
    - If a class is absent from both prediction and target, its score is 1.0
      (true negative).
    - If a class is predicted but absent from the target (or vice versa), its
      score is 0.0.

    Args:
        predictions: Integer class predictions of shape (...).
        targets: Integer class targets of shape (...).
        num_classes: Number of classes.
        epsilon: Small constant for numerical stability.

    Returns:
        Per-class Dice scores of shape (num_classes,).
    """
    intersection, prediction_area, target_area = _per_class_counts(
        predictions, targets, num_classes
    )
    dice = (2.0 * intersection + epsilon) / (prediction_area + target_area + epsilon)

    # True negatives: class absent from both prediction and target -> 1.0.
    absent_from_both = (prediction_area == 0) & (target_area == 0)
    return jnp.where(absent_from_both, 1.0, dice)


def iou_score(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    num_classes: int,
    *,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """Per-class intersection-over-union (Jaccard) score.

    Empty-class behavior matches :func:`dice_score`: true negatives score 1.0,
    false positives/negatives score 0.0.

    Args:
        predictions: Integer class predictions of shape (...).
        targets: Integer class targets of shape (...).
        num_classes: Number of classes.
        epsilon: Small constant for numerical stability.

    Returns:
        Per-class IoU scores of shape (num_classes,).
    """
    intersection, prediction_area, target_area = _per_class_counts(
        predictions, targets, num_classes
    )
    union = prediction_area + target_area - intersection
    iou = (intersection + epsilon) / (union + epsilon)

    absent_from_both = (prediction_area == 0) & (target_area == 0)
    return jnp.where(absent_from_both, 1.0, iou)


def segmentation_metrics(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    num_classes: int,
    *,
    epsilon: float = 1e-6,
) -> dict[str, jnp.ndarray]:
    """Compute per-class Dice and IoU in one call.

    Args:
        predictions: Integer class predictions of shape (...).
        targets: Integer class targets of shape (...).
        num_classes: Number of classes.
        epsilon: Small constant for numerical stability.

    Returns:
        Dictionary with keys ``dice`` and ``iou``.
    """
    return {
        "dice": dice_score(predictions, targets, num_classes, epsilon=epsilon),
        "iou": iou_score(predictions, targets, num_classes, epsilon=epsilon),
    }
