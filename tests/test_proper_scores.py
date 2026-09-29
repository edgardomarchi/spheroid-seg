"""Tests for proper scoring rules (multiclass Brier + log loss) in metrics.py.

Proper scoring rules (Gneiting & Raftery 2007, JASA 102(477)) evaluate the
softmax probabilities, not the argmax mask: they measure probabilistic quality
and calibration, which Dice/IoU cannot see. All functions take probabilities
(``(..., C)`` arrays whose last axis is a valid distribution) plus integer
targets of shape ``(...)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from spheroid_seg.metrics import (
    PROPER_SCORE_EPSILON,
    add_proper_score_sums,
    empty_proper_score_sums,
    finalize_proper_scores,
    log_loss,
    multiclass_brier_score,
    per_class_brier_score,
    per_class_log_loss,
    proper_score_partial_sums,
)

LN_3 = float(np.log(3.0))


def test_uniform_predictions_closed_form() -> None:
    """Uniform predictions (p = 1/3) match closed-form constants.

    log loss = ln(3), multiclass Brier = 2/3, per-class Brier = 2/9.
    """
    targets = np.tile(np.array([0, 1, 2], dtype=np.int64), 20)  # all classes present
    probs = np.full((60, 3), 1.0 / 3.0, dtype=np.float64)

    assert log_loss(probs, targets, 3) == pytest.approx(LN_3, rel=1e-12)
    assert multiclass_brier_score(probs, targets, 3) == pytest.approx(2.0 / 3.0, rel=1e-12)

    # One-vs-rest Brier for a class with GT prevalence 1/3:
    # (1/3)(1/3 - 1)^2 + (2/3)(1/3 - 0)^2 = 4/27 + 2/27 = 2/9.
    np.testing.assert_allclose(
        per_class_brier_score(probs, targets, 3),
        np.full(3, 2.0 / 9.0),
        rtol=1e-12,
    )
    # Conditional log loss within each class's GT region is -log(1/3) = ln 3.
    np.testing.assert_allclose(
        per_class_log_loss(probs, targets, 3),
        np.full(3, LN_3),
        rtol=1e-12,
    )


def test_perfect_predictions_zero_scores() -> None:
    """One-hot probabilities give exactly zero loss and zero Brier."""
    targets = np.array([0, 1, 2, 1], dtype=np.int64)
    probs = np.eye(3, dtype=np.float64)[targets]

    assert log_loss(probs, targets, 3) == 0.0
    assert multiclass_brier_score(probs, targets, 3) == 0.0
    np.testing.assert_array_equal(per_class_brier_score(probs, targets, 3), np.zeros(3))
    np.testing.assert_array_equal(per_class_log_loss(probs, targets, 3), np.zeros(3))


def test_near_perfect_predictions_approximately_zero() -> None:
    """p_true = 1 - 1e-9: both scores are ~0 (the epsilon clip must not fire here)."""
    targets = np.array([0, 2], dtype=np.int64)
    probs = np.array(
        [
            [1.0 - 1e-9, 5e-10, 5e-10],
            [1e-9, 1e-9, 1.0 - 2e-9],
        ],
        dtype=np.float64,
    )

    assert log_loss(probs, targets, 3) < 1e-6
    assert multiclass_brier_score(probs, targets, 3) < 1e-6


def test_epsilon_clip_bounds_log_loss_of_zero_probabilities() -> None:
    """A zero predicted probability for the true class clips to the documented epsilon.

    log(eps) = -16.118... for eps = 1e-7, so a fully confident wrong prediction
    costs at most -log(eps); the clip is deterministic.
    """
    targets = np.array([1], dtype=np.int64)
    probs = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    assert PROPER_SCORE_EPSILON == 1e-7
    assert log_loss(probs, targets, 3) == pytest.approx(-np.log(PROPER_SCORE_EPSILON))


def test_log_loss_matches_optax_softmax_cross_entropy() -> None:
    """Our log loss agrees with optax.softmax_cross_entropy on identical logits."""
    rng = np.random.default_rng(123)
    logits = rng.normal(size=(4, 8, 8, 3)).astype(np.float32)
    targets = rng.integers(0, 3, size=(4, 8, 8))

    probs = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
    ours = log_loss(probs, targets, 3)

    one_hot = jax.nn.one_hot(jnp.asarray(targets), num_classes=3)
    expected = float(np.mean(np.asarray(optax.softmax_cross_entropy(jnp.asarray(logits), one_hot))))

    assert ours == pytest.approx(expected, abs=1e-5)


def test_hand_computed_two_pixel_case() -> None:
    """Two-pixel case pins down per-class semantics exactly.

    Pixel A: GT 0, p = [0.8, 0.1, 0.1]; pixel B: GT 1, p = [0.2, 0.7, 0.1].
    """
    targets = np.array([0, 1], dtype=np.int64)
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
        ],
        dtype=np.float64,
    )

    # Multiclass Brier: per pixel sum_c (p_c - y_c)^2 = 0.06 + 0.14, mean = 0.10.
    assert multiclass_brier_score(probs, targets, 3) == pytest.approx(0.10)
    # One-vs-rest Brier per class, mean over both pixels.
    np.testing.assert_allclose(
        per_class_brier_score(probs, targets, 3),
        np.array([0.04, 0.05, 0.01]),
        rtol=1e-12,
    )
    # Overall log loss and per-class conditional log loss.
    assert log_loss(probs, targets, 3) == pytest.approx(-0.5 * (np.log(0.8) + np.log(0.7)))
    per_class = per_class_log_loss(probs, targets, 3)
    assert per_class[0] == pytest.approx(-np.log(0.8))
    assert per_class[1] == pytest.approx(-np.log(0.7))
    assert np.isnan(per_class[2])  # class 2 absent from GT


def test_absent_class_conditional_log_loss_is_nan() -> None:
    """A class entirely absent from GT yields NaN conditional log loss; the rest is unaffected."""
    targets = np.zeros((4, 4), dtype=np.int64)
    targets[:, 2] = 1  # classes 0 and 1 only
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(4, 4, 3))
    probs = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))

    scores = finalize_proper_scores(proper_score_partial_sums(probs, targets, 3), 3)

    assert np.isnan(scores["log_loss_per_class"][2])
    assert np.isfinite(scores["log_loss_per_class"][0])
    assert np.isfinite(scores["log_loss_per_class"][1])
    # Scalar scores are still defined over all pixels.
    assert np.isfinite(scores["brier"])
    assert np.isfinite(scores["log_loss"])
    # One-vs-rest Brier is a mean over all pixels, so it stays defined too.
    assert np.isfinite(scores["brier_per_class"][2])
    # The absent class contributed zero pixels to its conditional count.
    partial = proper_score_partial_sums(probs, targets, 3)
    assert partial["log_loss_per_class_count"][2] == 0
    assert partial["log_loss_per_class_count"][0] == 12
    assert partial["log_loss_per_class_count"][1] == 4


def test_partial_sums_streaming_matches_one_shot() -> None:
    """Accumulating partial sums over chunks equals the one-shot computation.

    This is the contract the streaming eval relies on: pooled scores are the
    exact float64 sums of per-tile partial sums, never per-pixel maps.
    """
    rng = np.random.default_rng(11)
    probs = np.asarray(jax.nn.softmax(jnp.asarray(rng.normal(size=(6, 5, 5, 3))), axis=-1))
    targets = rng.integers(0, 3, size=(6, 5, 5))

    total = empty_proper_score_sums(3)
    for chunk in range(0, 6, 2):
        part = proper_score_partial_sums(probs[chunk : chunk + 2], targets[chunk : chunk + 2], 3)
        add_proper_score_sums(total, part)
    streamed = finalize_proper_scores(total, 3)

    assert streamed["brier"] == pytest.approx(multiclass_brier_score(probs, targets, 3))
    assert streamed["log_loss"] == pytest.approx(log_loss(probs, targets, 3))
    np.testing.assert_allclose(
        streamed["brier_per_class"], per_class_brier_score(probs, targets, 3)
    )
    np.testing.assert_allclose(
        streamed["log_loss_per_class"], per_class_log_loss(probs, targets, 3)
    )


def test_float64_accumulation_avoids_float32_precision_loss() -> None:
    """Partial sums accumulate in float64: >2**24 identical pixels must not lose mass.

    Mirrors the confusion-matrix float32-saturation regression test: many small
    identical per-pixel contributions must sum exactly.
    """
    n = 40_000_000  # > 2**25; a float32 accumulator would round the running sum
    tile_n = 5_000_000
    p_nll = 0.1  # per-pixel -log(p_true) after clipping
    probs = np.full((tile_n, 3), 1e-7, dtype=np.float64)
    probs[:, 0] = np.exp(-p_nll)
    targets = np.zeros(tile_n, dtype=np.int64)

    total = empty_proper_score_sums(3)
    for _ in range(n // tile_n):
        add_proper_score_sums(total, proper_score_partial_sums(probs, targets, 3))
    scores = finalize_proper_scores(total, 3)

    assert scores["log_loss"] == pytest.approx(p_nll, rel=1e-12)
    assert total["log_loss_sum"] == pytest.approx(n * p_nll, rel=1e-12)


def test_invalid_inputs_are_rejected() -> None:
    """Shape mismatches, out-of-range targets, and non-integer targets raise ValueError."""
    probs = np.full((4, 3), 1.0 / 3.0)
    good_targets = np.zeros(4, dtype=np.int64)

    with pytest.raises(ValueError, match="shape"):
        log_loss(probs, np.zeros((4, 1), dtype=np.int64), 3)
    with pytest.raises(ValueError, match="num_classes"):
        multiclass_brier_score(np.full((4, 2), 0.5), good_targets, 3)
    with pytest.raises(ValueError, match=r"in \[0"):
        log_loss(probs, np.array([0, 1, 2, 3], dtype=np.int64), 3)
    with pytest.raises(ValueError, match="integer"):
        log_loss(probs, np.zeros(4, dtype=np.float64), 3)
