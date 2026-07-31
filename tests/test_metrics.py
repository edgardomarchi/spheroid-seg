"""Tests for per-class Dice and IoU metrics."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from spheroid_seg.metrics import dice_score, iou_score, segmentation_metrics


def test_perfect_prediction_metrics_are_one() -> None:
    """Perfect predictions yield Dice and IoU of 1.0 for every class."""
    targets = jnp.array([[[0, 1], [2, 0]]], dtype=jnp.int32)
    predictions = targets

    dice = dice_score(predictions, targets, num_classes=3)
    iou = iou_score(predictions, targets, num_classes=3)

    np.testing.assert_allclose(dice, jnp.ones(3), atol=1e-6)
    np.testing.assert_allclose(iou, jnp.ones(3), atol=1e-6)


def test_dice_score_hand_computed() -> None:
    """Dice on a tiny array matches the manual 2*|inter| / (|pred|+|target|)."""
    # Class 0 target: 3 pixels; class 1 target: 1 pixel.
    targets = jnp.array([[[0, 1], [0, 0]]], dtype=jnp.int32)
    # Predictions: class 0 everywhere.
    predictions = jnp.zeros_like(targets)

    dice = dice_score(predictions, targets, num_classes=2, epsilon=0.0)

    # Class 0: intersection 3, |pred|=4, |target|=3 -> 2*3/(4+3)=6/7
    # Class 1: intersection 0, |pred|=0, |target|=1 -> 0
    np.testing.assert_allclose(dice, jnp.array([6.0 / 7.0, 0.0]), atol=1e-6)


def test_iou_score_hand_computed() -> None:
    """IoU on a tiny array matches manual |inter| / |union|."""
    targets = jnp.array([[[0, 1], [0, 0]]], dtype=jnp.int32)
    predictions = jnp.zeros_like(targets)

    iou = iou_score(predictions, targets, num_classes=2, epsilon=0.0)

    # Class 0: inter=3, union=4 -> 3/4
    # Class 1: inter=0, union=1 -> 0
    np.testing.assert_allclose(iou, jnp.array([3.0 / 4.0, 0.0]), atol=1e-6)


def test_empty_class_true_negative() -> None:
    """A class absent from both prediction and target scores 1.0."""
    targets = jnp.zeros((1, 4, 4), dtype=jnp.int32)
    predictions = jnp.zeros_like(targets)

    dice = dice_score(predictions, targets, num_classes=2)
    iou = iou_score(predictions, targets, num_classes=2)

    np.testing.assert_allclose(dice, jnp.ones(2), atol=1e-6)
    np.testing.assert_allclose(iou, jnp.ones(2), atol=1e-6)


def test_empty_class_false_positive() -> None:
    """A class predicted but not present in the target scores 0.0."""
    targets = jnp.zeros((1, 4, 4), dtype=jnp.int32)
    predictions = jnp.ones_like(targets)

    dice = dice_score(predictions, targets, num_classes=2)
    iou = iou_score(predictions, targets, num_classes=2)

    assert dice[0] == pytest.approx(0.0, abs=1e-6)
    assert dice[1] == pytest.approx(0.0, abs=1e-6)
    assert iou[0] == pytest.approx(0.0, abs=1e-6)
    assert iou[1] == pytest.approx(0.0, abs=1e-6)


def test_segmentation_metrics_returns_both() -> None:
    """segmentation_metrics returns per-class Dice and IoU arrays."""
    targets = jnp.array([[[0, 1], [2, 0]]], dtype=jnp.int32)
    metrics = segmentation_metrics(targets, targets, num_classes=3)

    assert "dice" in metrics
    assert "iou" in metrics
    assert metrics["dice"].shape == (3,)
    assert metrics["iou"].shape == (3,)
