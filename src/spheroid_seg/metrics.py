"""Per-class segmentation metrics: Dice, IoU, and proper scoring rules."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# Deterministic lower clip for predicted probabilities before taking logs.
# A fully confident wrong prediction therefore costs at most -log(1e-7), and
# log(0) can never produce -inf/NaN. Fixed constant by design (no config key):
# proper scores must be comparable across runs, configs, and checkpoints.
PROPER_SCORE_EPSILON = 1e-7


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


def _validate_prob_inputs(
    probs: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> None:
    """Validate the probability/target input convention shared by proper scores.

    Convention: ``probs`` has shape ``(..., num_classes)`` with a probability
    distribution along the last axis (softmax output, not logits); ``targets``
    has shape ``(...)`` with integer class IDs in ``[0, num_classes)``.
    """
    if probs.ndim < 1 or probs.shape[:-1] != targets.shape:
        raise ValueError(
            f"probs shape {probs.shape} is incompatible with targets shape {targets.shape}"
        )
    if probs.shape[-1] != num_classes:
        raise ValueError(f"probs last axis {probs.shape[-1]} != num_classes {num_classes}")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError(f"targets must be an integer array, got dtype {targets.dtype}")
    if np.any((targets < 0) | (targets >= num_classes)):
        raise ValueError(f"targets must be in [0, {num_classes})")


def proper_score_partial_sums(
    probs: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    *,
    epsilon: float = PROPER_SCORE_EPSILON,
) -> dict[str, np.ndarray]:
    """Accumulable per-batch statistics for the proper scoring rules.

    All sums are accumulated in float64 (NumPy) so pooled scores stay exact over
    hundreds of millions of pixels, mirroring the uint32 care taken for the
    confusion matrix; counts are exact int64. The returned dictionary is designed
    to be summed across tiles/images with :func:`add_proper_score_sums` and
    reduced once with :func:`finalize_proper_scores`, so eval never retains
    per-pixel probability maps beyond the current tile.

    Args:
        probs: Probability array of shape ``(..., num_classes)`` (softmax
            output, not logits).
        targets: Integer targets of shape ``(...)`` in ``[0, num_classes)``.
        num_classes: Number of classes.
        epsilon: Deterministic lower clip applied to probabilities before
            ``log`` (see :data:`PROPER_SCORE_EPSILON`).

    Returns:
        Dictionary with keys ``brier_sum``, ``brier_per_class_sum`` (float64,
        shape ``(num_classes,)``), ``log_loss_sum``, ``log_loss_per_class_sum``
        (float64, shape ``(num_classes,)``), ``log_loss_per_class_count``
        (int64, shape ``(num_classes,)``), and ``n_pixels`` (int64).
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets)
    _validate_prob_inputs(probs, targets, num_classes)
    targets = targets.astype(np.intp, copy=False)

    one_hot = np.eye(num_classes, dtype=np.float64)[targets]
    diff = probs - one_hot
    brier_per_class_sum = np.sum(diff * diff, axis=tuple(range(probs.ndim - 1)), dtype=np.float64)
    brier_sum = np.array(np.sum(brier_per_class_sum), dtype=np.float64)

    p_true = np.take_along_axis(probs, targets[..., None], axis=-1)[..., 0]
    nll = -np.log(np.clip(p_true, epsilon, 1.0))
    log_loss_per_class_sum = np.zeros(num_classes, dtype=np.float64)
    np.add.at(log_loss_per_class_sum, targets.ravel(), nll.ravel())
    log_loss_sum = np.array(np.sum(nll), dtype=np.float64)

    return {
        "brier_sum": brier_sum,
        "brier_per_class_sum": brier_per_class_sum,
        "log_loss_sum": log_loss_sum,
        "log_loss_per_class_sum": log_loss_per_class_sum,
        "log_loss_per_class_count": np.bincount(targets.ravel(), minlength=num_classes).astype(
            np.int64
        ),
        "n_pixels": np.array(targets.size, dtype=np.int64),
    }


def empty_proper_score_sums(num_classes: int) -> dict[str, np.ndarray]:
    """Zero-initialized accumulator compatible with :func:`proper_score_partial_sums`."""
    return {
        "brier_sum": np.float64(0.0),
        "brier_per_class_sum": np.zeros(num_classes, dtype=np.float64),
        "log_loss_sum": np.float64(0.0),
        "log_loss_per_class_sum": np.zeros(num_classes, dtype=np.float64),
        "log_loss_per_class_count": np.zeros(num_classes, dtype=np.int64),
        "n_pixels": np.int64(0),
    }


def add_proper_score_sums(
    total: dict[str, np.ndarray],
    part: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Add a partial-sums dictionary into ``total`` in place and return ``total``."""
    for key, value in part.items():
        total[key] = total[key] + value
    return total


def finalize_proper_scores(
    partial: dict[str, Any],
    num_classes: int,
) -> dict[str, Any]:
    """Reduce accumulated partial sums to final proper scoring rules.

    The per-class conditional log loss of a class absent from the ground truth
    is undefined: it is reported as NaN (never a made-up convention).

    Args:
        partial: Accumulated dictionary from :func:`proper_score_partial_sums`.
        num_classes: Number of classes.

    Returns:
        Dictionary with keys ``brier`` (multiclass, float), ``brier_per_class``
        (one-vs-rest, shape ``(num_classes,)``), ``log_loss`` (float), and
        ``log_loss_per_class`` (conditional on GT class, shape
        ``(num_classes,)`` with NaN for absent classes).
    """
    counts = partial["log_loss_per_class_count"]
    with np.errstate(invalid="ignore", divide="ignore"):
        log_loss_per_class = np.where(
            counts > 0,
            partial["log_loss_per_class_sum"] / np.maximum(counts, 1),
            np.nan,
        )
        n_pixels = max(int(partial["n_pixels"]), 1)
        return {
            "brier": float(partial["brier_sum"] / n_pixels),
            "brier_per_class": partial["brier_per_class_sum"] / n_pixels,
            "log_loss": float(partial["log_loss_sum"] / n_pixels),
            "log_loss_per_class": log_loss_per_class,
        }


def _proper_scores(
    probs: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    *,
    epsilon: float = PROPER_SCORE_EPSILON,
) -> dict[str, Any]:
    """One-shot proper scores: partial sums over a single array, then finalize."""
    return finalize_proper_scores(
        proper_score_partial_sums(probs, targets, num_classes, epsilon=epsilon),
        num_classes,
    )


def multiclass_brier_score(
    probs: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> float:
    """Multiclass Brier score: mean over pixels of ``sum_c (p_c - y_c)^2``.

    Args:
        probs: Probability array of shape ``(..., num_classes)`` (softmax
            output, not logits).
        targets: Integer targets of shape ``(...)`` in ``[0, num_classes)``.
        num_classes: Number of classes.

    Returns:
        Scalar mean in ``[0, 2]``; lower is better, 0 is perfect.
    """
    return float(_proper_scores(probs, targets, num_classes)["brier"])


def per_class_brier_score(
    probs: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Per-class one-vs-rest Brier score: mean over pixels of ``(p_c - y_c)^2``.

    Defined for every class regardless of presence (the mean runs over all
    pixels; an absent class scores ``mean(p_c^2)``).

    Returns:
        Array of shape ``(num_classes,)``; lower is better, 0 is perfect.
    """
    return _proper_scores(probs, targets, num_classes)["brier_per_class"]


def log_loss(
    probs: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    *,
    epsilon: float = PROPER_SCORE_EPSILON,
) -> float:
    """Negative log-likelihood: ``-mean(log p_true)`` over all pixels.

    Probabilities are clipped to ``[epsilon, 1]`` before the log (deterministic;
    see :data:`PROPER_SCORE_EPSILON`).

    Returns:
        Scalar mean; lower is better, 0 is perfect.
    """
    return float(_proper_scores(probs, targets, num_classes, epsilon=epsilon)["log_loss"])


def per_class_log_loss(
    probs: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    *,
    epsilon: float = PROPER_SCORE_EPSILON,
) -> np.ndarray:
    """Per-class conditional log loss: ``-mean(log p_true)`` on pixels with GT class c.

    This measures calibration within each class's ground-truth region. A class
    absent from the ground truth has no such pixels: the value is NaN (never a
    made-up convention).

    Returns:
        Array of shape ``(num_classes,)``; lower is better, 0 is perfect.
    """
    return _proper_scores(probs, targets, num_classes, epsilon=epsilon)["log_loss_per_class"]
