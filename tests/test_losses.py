"""Tests for the combined Dice + cross-entropy loss."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from spheroid_seg.losses import (
    cross_entropy_loss,
    dice_loss,
    segmentation_loss,
)


def _one_hot_logits(targets: jnp.ndarray, num_classes: int, big: float = 1e6) -> jnp.ndarray:
    """Create near-perfect logits from integer targets."""
    logits = jnp.full((*targets.shape, num_classes), -big, dtype=jnp.float32)
    for c in range(num_classes):
        logits = logits.at[..., c].set(jnp.where(targets == c, big, -big))
    return logits


def test_perfect_prediction_loss_is_near_zero() -> None:
    """A perfect prediction should give a loss very close to zero."""
    targets = jnp.array([[[0, 1], [2, 0]]], dtype=jnp.int32)
    logits = _one_hot_logits(targets, num_classes=3)

    total = segmentation_loss(logits, targets)
    assert total < 1e-5


def test_dice_loss_hand_computed() -> None:
    """Soft Dice on a tiny array matches a manual calculation."""
    # Two classes, 1x2x2.
    # Class 0: target pixels at (0,0) and (0,1) -> 2 of 4
    # Class 1: target pixel at (1,0) only -> 1 of 4
    targets = jnp.array([[[0, 1], [0, 0]]], dtype=jnp.int32)
    # Logits strongly favor class 0 everywhere -> p0≈1, p1≈0.
    logits = jnp.array(
        [[[[10.0, -10.0], [10.0, -10.0]], [[10.0, -10.0], [10.0, -10.0]]]],
        dtype=jnp.float32,
    )

    loss = dice_loss(logits, targets, epsilon=0.0)

    # Softmax -> p0 ≈ 1, p1 ≈ 0.
    # Class 0 target: 3 pixels; class 1 target: 1 pixel.
    # Class 0: intersection = 3, denom = sum(p0^2 + t0^2) = 4 + 3 = 7
    #   dice_0 = 2*3/7 = 6/7
    # Class 1: intersection = 0, denom = 0 + 1 = 1
    #   dice_1 = 0
    # mean dice = (6/7 + 0) / 2 = 3/7, loss = 1 - 3/7 = 4/7
    np.testing.assert_allclose(float(loss), 4.0 / 7.0, atol=1e-6)


def test_cross_entropy_loss_hand_computed() -> None:
    """Cross-entropy on a tiny array matches a manual softmax calculation."""
    logits = jnp.array(
        [[[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]],
        dtype=jnp.float32,
    )
    targets = jnp.array([[[0, 0], [0, 1]]], dtype=jnp.int32)

    loss = cross_entropy_loss(logits, targets)

    # Pixel (0,0): target 0, logits [0,0] -> -log_softmax0 = log(2)
    # Pixel (0,1): target 0, logits [0,0] -> log(2)
    # Pixel (1,0): target 0, logits [1,0] -> -log_softmax0 = log(1+e) - 1
    # Pixel (1,1): target 1, logits [0,1] -> -log_softmax1 = log(1+e) - 1
    expected = (2 * jnp.log(2.0) + 2 * (jnp.log(1 + jnp.e) - 1.0)) / 4.0
    np.testing.assert_allclose(float(loss), float(expected), atol=1e-6)


def test_segmentation_loss_is_weighted_sum() -> None:
    """The combined loss equals dice_weight*Dice + ce_weight*CE."""
    targets = jnp.array([[[0, 1], [2, 0]]], dtype=jnp.int32)
    logits = jnp.array(
        np.random.default_rng(0).standard_normal((*targets.shape, 3)),
        dtype=jnp.float32,
    )

    dice = dice_loss(logits, targets)
    ce = cross_entropy_loss(logits, targets)
    total = segmentation_loss(logits, targets, dice_weight=0.7, ce_weight=0.3)

    np.testing.assert_allclose(float(total), 0.7 * float(dice) + 0.3 * float(ce), atol=1e-6)


def test_class_weights_downweight_background() -> None:
    """Down-weighting background reduces CE when background is correctly predicted."""
    # Two classes; all pixels are background (0) and logits weakly favor 0.
    targets = jnp.zeros((1, 4, 4), dtype=jnp.int32)
    logits = jnp.full((1, 4, 4, 2), -2.0, dtype=jnp.float32)
    logits = logits.at[..., 0].set(2.0)

    unweighted = cross_entropy_loss(logits, targets)
    downweighted = cross_entropy_loss(logits, targets, class_weights=jnp.array([0.1, 1.0]))

    assert downweighted < unweighted
