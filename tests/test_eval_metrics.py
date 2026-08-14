"""Tests for evaluation metric accumulation and grouping."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from spheroid_seg.eval import (
    accumulate_confusion_matrix,
    class_metrics_from_confusion,
    group_by_magnification,
)
from spheroid_seg.metrics import dice_score, iou_score

CLASS_NAMES = ["background", "loose cell", "aggregate"]


def test_hand_computed_confusion_matrix() -> None:
    """Confusion matrix matches a manually checked 3-class case."""
    # Class 0: 3, class 1: 1, class 2: 0 in target.
    # Predicted: class 0 everywhere.
    target = jnp.array([[0, 1], [0, 0]], dtype=jnp.int32)
    pred = jnp.zeros_like(target)

    conf = accumulate_confusion_matrix(pred, target, num_classes=3)
    expected = np.array(
        [
            [3, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(np.asarray(conf), expected)


def test_pooled_dice_matches_metrics_score() -> None:
    """Dice recomputed from the global confusion matrix equals per-pixel metrics."""
    rng = np.random.default_rng(7)
    pred = rng.integers(0, 3, size=(10, 32, 32), dtype=np.int32)
    target = rng.integers(0, 3, size=(10, 32, 32), dtype=jnp.int32)

    conf = accumulate_confusion_matrix(pred, target, num_classes=3)
    pooled = class_metrics_from_confusion(conf)

    dice = dice_score(pred, target, num_classes=3)
    iou = iou_score(pred, target, num_classes=3)

    np.testing.assert_allclose(np.asarray(pooled["dice"]), np.asarray(dice), atol=1e-5)
    np.testing.assert_allclose(np.asarray(pooled["iou"]), np.asarray(iou), atol=1e-5)


def test_empty_class_true_negative_in_confusion() -> None:
    """A class absent from both prediction and target scores 1.0 when pooled."""
    target = jnp.zeros((4, 4), dtype=jnp.int32)
    pred = jnp.zeros_like(target)

    conf = accumulate_confusion_matrix(pred, target, num_classes=3)
    pooled = class_metrics_from_confusion(conf)

    np.testing.assert_allclose(np.asarray(pooled["dice"]), np.ones(3), atol=1e-6)
    np.testing.assert_allclose(np.asarray(pooled["iou"]), np.ones(3), atol=1e-6)


def test_false_positive_negative_score_zero() -> None:
    """A class predicted but not present (or vice versa) scores 0.0."""
    target = jnp.zeros((4, 4), dtype=jnp.int32)
    pred = jnp.ones_like(target)

    conf = accumulate_confusion_matrix(pred, target, num_classes=3)
    pooled = class_metrics_from_confusion(conf)

    assert pooled["dice"][0] == pytest.approx(0.0, abs=1e-6)
    assert pooled["dice"][1] == pytest.approx(0.0, abs=1e-6)


def test_group_by_magnification() -> None:
    """Images are grouped by filename suffix."""
    names = ["a_4x", "b_4x", "c_10x", "d_unknown"]
    values = [1, 2, 3, 4]
    grouped = group_by_magnification(names, values)

    assert set(grouped.keys()) == {"4x", "10x", "unknown"}
    assert grouped["4x"] == [1, 2]
    assert grouped["10x"] == [3]
    assert grouped["unknown"] == [4]


def test_accumulate_confusion_matrix_avoids_float32_saturation() -> None:
    """Regression: counts must stay exact past the float32 mantissa (2**24).

    The old path built float32 one-hot vectors and used jnp.dot, whose
    accumulator rounds to the nearest representable float32. Beyond 2**24
    pixels of a single class, adding 1.0 rounds back to the same value, so
    the count freezes at 16,777,216. This test accumulates >2**24 pixels of
    class 0 and asserts the exact integer count.
    """
    n_tiles = 5
    tile = 2048
    total_pixels = n_tiles * tile * tile  # 20_971_520 > 2**24

    target = jnp.zeros((n_tiles, tile, tile), dtype=jnp.int32)
    pred = jnp.zeros_like(target)

    conf = accumulate_confusion_matrix(pred, target, num_classes=3)
    conf = np.asarray(conf)

    assert conf.dtype == np.uint32
    assert conf[0, 0] == total_pixels
    assert conf.sum() == total_pixels
