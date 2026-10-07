"""Unit tests for the validation diagnostics module.

Covers the three analyses (patch/full-image prevalence, false-positive
connected components, background-logit-bias sweep), CLI argument parsing
helpers, and the documented acceptance invariants:

- bias ``0`` reproduces the existing eval counting semantics exactly;
- predicted foreground pixels are monotonic non-increasing in the bias;
- component areas sum exactly to the false-positive pixel count;
- reflect-padded pixels are excluded from every sweep accumulator.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import numpy as np
import pytest

from spheroid_seg.eval import _tile_valid_slices
from spheroid_seg.metrics import proper_score_partial_sums
from spheroid_seg.validation_diagnostics import (
    ANALYSIS_MASKS,
    DEFAULT_AREA_THRESHOLDS,
    DEFAULT_BACKGROUND_BIAS_GRID,
    BiasGridAccumulators,
    adjusted_prediction,
    class_pixel_counts,
    component_summary,
    component_table,
    false_positive_mask,
    full_image_prevalence,
    object_precision_recall_from_3x3,
    parse_area_thresholds,
    parse_bias_grid,
    precision_recall_from_confusion,
    saved_patch_prevalence,
    sweep_full_image,
)

CLASS_MAPPING = {0: 0, 1: 1, 2: 2, 3: 2}


# ---------------------------------------------------------------------------
# Part 1 — class prevalence
# ---------------------------------------------------------------------------


def test_class_pixel_counts_merge_ids_2_and_3() -> None:
    """IDs 2 and 3 are counted together as model class 2 (aggregate)."""
    mask = np.array([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=np.uint8)
    counts = class_pixel_counts(mask, 3, CLASS_MAPPING)
    assert counts.dtype == np.uint64
    assert counts.tolist() == [2, 2, 4]


def test_class_pixel_counts_identity_mapping() -> None:
    """Without a mapping the raw IDs are counted as-is."""
    mask = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    counts = class_pixel_counts(mask, 3)
    assert counts.tolist() == [2, 2, 2]


def test_class_pixel_counts_rejects_out_of_range_ids() -> None:
    """Mask values outside [0, num_classes) are a schema error, never silent."""
    mask = np.array([[0, 1, 3]], dtype=np.uint8)
    with pytest.raises(ValueError, match="class ID"):
        class_pixel_counts(mask, 3)


def _write_mask(masks_dir: Path, name: str, mask: np.ndarray) -> None:
    import cv2

    masks_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(masks_dir / f"{name}.png"), mask)


def _config_for_prevalence(
    tmp_path: Path, raw_dir: Path, masks_dir: Path, splits_dir: Path
) -> dict:
    import yaml

    config = yaml.safe_load(Path("configs/tiny.yaml").read_text())
    config["data"] = {
        "raw_dir": str(raw_dir),
        "masks_dir": str(masks_dir),
        "splits_dir": str(splits_dir),
        "slimia_dir": str(tmp_path / "slimia"),
    }
    return config


def test_full_image_prevalence_equals_sum_of_per_image_counts(tmp_path: Path) -> None:
    """Pooled full-image counts are the exact sum of the per-image counts."""
    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    mask_a = np.array([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=np.uint8)
    mask_b = np.array([[1, 1, 0, 0], [2, 2, 3, 3]], dtype=np.uint8)
    _write_mask(masks_dir, "img_a_4x", mask_a)
    _write_mask(masks_dir, "img_b_10x", mask_b)
    # Raw files are not needed for prevalence but make the pair real.
    _write_mask(raw_dir, "img_a_4x", mask_a)
    _write_mask(raw_dir, "img_b_10x", mask_b)
    (splits_dir / "train.txt").write_text("img_a_4x\nimg_b_10x\n")

    config = _config_for_prevalence(tmp_path, raw_dir, masks_dir, splits_dir)
    prevalence = full_image_prevalence(config, splits=("train",), tmp_dir=tmp_path / "tmp")

    expected = class_pixel_counts(mask_a, 3, CLASS_MAPPING) + class_pixel_counts(
        mask_b, 3, CLASS_MAPPING
    )
    assert prevalence["train"]["counts"].tolist() == expected.tolist()
    assert prevalence["train"]["n_pixels"] == mask_a.size + mask_b.size


def test_saved_patch_prevalence_exact_counts(tmp_path: Path) -> None:
    """A synthetic training_patches.npz yields exact per-split class counts."""
    train_masks = np.stack(
        [
            np.full((4, 4), 0, dtype=np.int32),
            np.full((4, 4), 1, dtype=np.int32),
            np.full((4, 4), 2, dtype=np.int32),
        ]
    )
    val_masks = np.stack(
        [
            np.full((4, 4), 0, dtype=np.int32),
            np.full((4, 4), 2, dtype=np.int32),
        ]
    )
    npz_path = tmp_path / "training_patches.npz"
    np.savez(
        npz_path,
        train_images=np.zeros((3, 4, 4, 1), dtype=np.float32),
        train_masks=train_masks,
        val_images=np.zeros((2, 4, 4, 1), dtype=np.float32),
        val_masks=val_masks,
    )

    prevalence = saved_patch_prevalence(npz_path, num_classes=3)
    assert prevalence["train"]["counts"].tolist() == [16, 16, 16]
    assert prevalence["val"]["counts"].tolist() == [16, 0, 16]
    assert prevalence["train"]["n_pixels"] == 3 * 16
    assert prevalence["val"]["n_patches"] == 2


def test_saved_patch_prevalence_missing_file(tmp_path: Path) -> None:
    """A missing training_patches.npz fails with a clear error."""
    with pytest.raises(FileNotFoundError, match="training_patches.npz"):
        saved_patch_prevalence(tmp_path / "training_patches.npz", num_classes=3)


@pytest.mark.parametrize(
    "arrays",
    [
        pytest.param(
            {
                "train_images": np.zeros((2, 4, 4, 1), np.float32),
                "train_masks": np.zeros((2, 4, 4), np.int32),
                "val_images": np.zeros((1, 4, 4, 1), np.float32),
            },
            id="missing-key",
        ),
        pytest.param(
            {
                "train_images": np.zeros((2, 4, 4, 1), np.float32),
                "train_masks": np.zeros((2, 4, 4), np.float32),
                "val_images": np.zeros((1, 4, 4, 1), np.float32),
                "val_masks": np.zeros((1, 4, 4), np.float32),
            },
            id="float-masks",
        ),
        pytest.param(
            {
                "train_images": np.zeros((2, 4, 4, 1), np.float32),
                "train_masks": np.zeros((2, 4, 4), np.int32),
                "val_images": np.zeros((1, 4, 4, 1), np.float32),
                "val_masks": np.full((1, 4, 4), 3, np.int32),
            },
            id="out-of-range-values",
        ),
        pytest.param(
            {
                "train_images": np.zeros((2, 4, 4, 1), np.float32),
                "train_masks": np.zeros((2, 4, 4, 1), np.int32),
                "val_images": np.zeros((1, 4, 4, 1), np.float32),
                "val_masks": np.zeros((1, 4, 4), np.int32),
            },
            id="wrong-ndim",
        ),
        pytest.param(
            {
                "train_images": np.zeros((3, 4, 4, 1), np.float32),
                "train_masks": np.zeros((2, 4, 4), np.int32),
                "val_images": np.zeros((1, 4, 4, 1), np.float32),
                "val_masks": np.zeros((1, 4, 4), np.int32),
            },
            id="length-mismatch",
        ),
    ],
)
def test_saved_patch_prevalence_malformed_npz(tmp_path: Path, arrays: dict) -> None:
    """An NPZ whose schema cannot be interpreted safely is rejected clearly."""
    npz_path = tmp_path / "training_patches.npz"
    np.savez(npz_path, **arrays)
    with pytest.raises(ValueError, match="training_patches.npz"):
        saved_patch_prevalence(npz_path, num_classes=3)


# ---------------------------------------------------------------------------
# Part 2 — false-positive connected components
# ---------------------------------------------------------------------------


def test_false_positive_mask_definitions() -> None:
    """Object, loose-cell, and aggregate FP masks obey their definitions."""
    pred = np.array([[0, 1, 2, 1], [2, 2, 0, 0]], dtype=np.uint8)
    gt = np.array([[0, 0, 1, 1], [2, 0, 0, 1]], dtype=np.uint8)

    object_fp = false_positive_mask(pred, gt, "object")
    loose_fp = false_positive_mask(pred, gt, "loose cell")
    agg_fp = false_positive_mask(pred, gt, "aggregate")

    np.testing.assert_array_equal(
        object_fp, np.array([[False, True, False, False], [False, True, False, False]])
    )
    np.testing.assert_array_equal(
        loose_fp, np.array([[False, True, False, False], [False, False, False, False]])
    )
    np.testing.assert_array_equal(
        agg_fp, np.array([[False, False, True, False], [False, True, False, False]])
    )

    with pytest.raises(ValueError, match="analysis"):
        false_positive_mask(pred, gt, "nonsense")


def test_component_table_known_components() -> None:
    """A synthetic FP mask yields the expected count, areas, and geometry."""
    fp = np.zeros((20, 20), dtype=bool)
    fp[2:4, 2:4] = True  # area 4, bbox rows 2-3, cols 2-3
    fp[10:13, 15:20] = True  # area 15, touches the right image border
    image = np.full((20, 20), 0.5, dtype=np.float32)

    rows = component_table(fp, image)

    assert len(rows) == 2
    by_area = sorted(rows, key=lambda r: r["area"])
    small, large = by_area
    assert small["area"] == 4
    assert (small["bbox_min_row"], small["bbox_min_col"]) == (2, 2)
    assert (small["bbox_max_row"], small["bbox_max_col"]) == (3, 3)
    assert small["centroid_row"] == pytest.approx(2.5)
    assert small["centroid_col"] == pytest.approx(2.5)
    assert small["min_border_distance"] == 2
    assert small["touches_border"] is False
    assert small["mean_intensity"] == pytest.approx(0.5)
    assert small["median_intensity"] == pytest.approx(0.5)

    assert large["area"] == 15
    assert large["min_border_distance"] == 0
    assert large["touches_border"] is True
    # Component IDs are 1-based and unique.
    assert {r["component_id"] for r in rows} == {1, 2}


def test_component_table_8_connectivity() -> None:
    """Diagonal pixels form a single component (8-connectivity, not 4)."""
    fp = np.eye(4, dtype=bool)
    image = np.zeros((4, 4), dtype=np.float32)
    rows = component_table(fp, image)
    assert len(rows) == 1
    assert rows[0]["area"] == 4


def test_component_table_intensity_statistics() -> None:
    """Mean/median intensity use the normalized image inside the component."""
    fp = np.zeros((6, 6), dtype=bool)
    fp[1:3, 1:3] = True
    image = np.arange(36, dtype=np.float32).reshape(6, 6) / 35.0
    rows = component_table(fp, image)
    (row,) = rows
    values = image[1:3, 1:3].ravel()
    assert row["mean_intensity"] == pytest.approx(float(np.mean(values)))
    assert row["median_intensity"] == pytest.approx(float(np.median(values)))


def test_component_areas_sum_to_fp_pixel_count() -> None:
    """Acceptance invariant: component areas sum exactly to the FP count."""
    rng = np.random.default_rng(0)
    fp = rng.random((64, 48)) > 0.92
    image = rng.random((64, 48)).astype(np.float32)
    rows = component_table(fp, image)
    total = sum(int(r["area"]) for r in rows)
    assert total == int(fp.sum())


def test_component_min_border_distance() -> None:
    """Minimum distance to the image border is computed per component."""
    fp = np.zeros((20, 30), dtype=bool)
    fp[0, 10] = True  # top border
    fp[19, 5] = True  # bottom border
    fp[5, 0] = True  # left border
    fp[7, 29] = True  # right border
    fp[10:12, 10:12] = True  # interior 2x2 block: distance min(10, 10, 19-11, 29-11) = 8
    image = np.zeros((20, 30), dtype=np.float32)
    rows = component_table(fp, image)
    border_rows = [r for r in rows if r["touches_border"]]
    interior_rows = [r for r in rows if not r["touches_border"]]
    assert len(border_rows) == 4
    assert all(r["min_border_distance"] == 0 for r in border_rows)
    (interior,) = interior_rows
    assert interior["min_border_distance"] == 8
    assert interior["area"] == 4


def test_component_summary_thresholds_and_border() -> None:
    """Summary aggregates area thresholds and border fractions exactly."""
    fp = np.zeros((30, 30), dtype=bool)
    fp[5:7, 5:7] = True  # area 4, interior
    fp[0:3, 0:3] = True  # area 9, touches top/left border
    fp[10:20, 10:20] = True  # area 100, interior
    image = np.zeros((30, 30), dtype=np.float32)
    rows = component_table(fp, image)
    assert sorted(r["area"] for r in rows) == [4, 9, 100]
    summary = component_summary(rows, (16, 64))

    assert summary["total_fp_area"] == 113
    assert summary["n_components"] == 3
    assert summary["mean_area"] == pytest.approx(113 / 3)
    assert summary["median_area"] == pytest.approx(9.0)
    assert summary["max_area"] == 100
    assert summary["area_below_16"] == 13
    assert summary["frac_below_16"] == pytest.approx(13 / 113)
    assert summary["area_below_64"] == 13
    assert summary["frac_below_64"] == pytest.approx(13 / 113)
    assert summary["border_fp_area"] == 9
    assert summary["border_fp_fraction"] == pytest.approx(9 / 113)


def test_component_summary_empty() -> None:
    """An image/analysis mask with no false positives summarizes to zeros."""
    summary = component_summary([], DEFAULT_AREA_THRESHOLDS)
    assert summary["total_fp_area"] == 0
    assert summary["n_components"] == 0
    for key, value in summary.items():
        if key.startswith("frac_"):
            assert value == 0.0


# ---------------------------------------------------------------------------
# Part 3 — background-logit-bias sweep
# ---------------------------------------------------------------------------


def test_parse_bias_grid_defaults_and_whitespace() -> None:
    """The documented default grid parses; whitespace is tolerated."""
    assert parse_bias_grid("0,0.25,0.5,0.75,1,1.5,2") == DEFAULT_BACKGROUND_BIAS_GRID
    assert parse_bias_grid(" 0 , 1.5 ,2 ") == (0.0, 1.5, 2.0)
    assert parse_bias_grid("-0.5") == (-0.5,)


@pytest.mark.parametrize("text", ["", "   ", "0,abc", "nan", "0,NaN", "inf", "0,inf", "0,0", "1,1"])
def test_parse_bias_grid_invalid(text: str) -> None:
    """Empty, non-numeric, non-finite, and duplicate grids are rejected."""
    with pytest.raises(ValueError, match="bias"):
        parse_bias_grid(text)


def test_parse_area_thresholds_defaults() -> None:
    """The documented default thresholds parse exactly."""
    assert parse_area_thresholds("16,64,256,1024,4096") == DEFAULT_AREA_THRESHOLDS


@pytest.mark.parametrize("text", ["", "0", "-4", "3.5", "16,16", "8,abc", "1e3"])
def test_parse_area_thresholds_invalid(text: str) -> None:
    """Non-integer, non-positive, duplicate, or malformed thresholds fail."""
    with pytest.raises(ValueError, match="threshold"):
        parse_area_thresholds(text)


def test_adjusted_prediction_bias_zero_is_identity() -> None:
    """Bias 0 returns the original probabilities and argmax (exact-equality path)."""
    probs = np.array([[[0.4, 0.35, 0.25], [0.1, 0.6, 0.3]]], dtype=np.float32)
    pred, adjusted = adjusted_prediction(probs, 0.0)
    np.testing.assert_array_equal(pred, np.argmax(probs, axis=-1))
    np.testing.assert_array_equal(adjusted, probs)


def test_adjusted_prediction_direction_hand_computed() -> None:
    """A background bias shifts the argmax toward background only."""
    probs = np.array([[[0.3, 0.5, 0.2]]], dtype=np.float64)

    pred0, adj0 = adjusted_prediction(probs, 0.0)
    assert pred0[0, 0] == 1
    np.testing.assert_allclose(adj0.sum(axis=-1), 1.0, rtol=1e-12)

    # log(0.3) + 0.25 = -0.954 < log(0.5) = -0.693: foreground still wins.
    pred_small, _ = adjusted_prediction(probs, 0.25)
    assert pred_small[0, 0] == 1

    # log(0.3) + 1.0 = -0.204 > log(0.5): background wins.
    pred_large, adj_large = adjusted_prediction(probs, 1.0)
    assert pred_large[0, 0] == 0
    # softmax(log(p) + 1 * e_bg): p0' = 0.3*e / (0.3*e + 0.7).
    expected_bg = 0.3 * math.e / (0.3 * math.e + 0.7)
    assert adj_large[0, 0, 0] == pytest.approx(expected_bg, rel=1e-12)
    np.testing.assert_allclose(adj_large.sum(axis=-1), 1.0, rtol=1e-12)


def test_adjusted_prediction_monotonic_foreground_count() -> None:
    """Increasing the background bias never increases predicted foreground pixels.

    Float64 probabilities are used so every background probability is strictly
    positive; float32 softmax outputs can underflow to exactly 0, which no
    finite bias can revive (exact softmax semantics, documented in
    ``adjusted_prediction``).
    """
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(3), size=(64, 64))
    grid = np.linspace(0.0, 50.0, 13)
    fg_counts = []
    for delta in grid:
        pred, _ = adjusted_prediction(probs, float(delta))
        fg_counts.append(int(np.count_nonzero(pred >= 1)))
    assert all(later <= earlier for earlier, later in zip(fg_counts, fg_counts[1:], strict=False))
    assert fg_counts[-1] == 0  # a large enough bias removes all foreground


def test_adjusted_prediction_huge_bias_all_background() -> None:
    """A very large positive background bias predicts background everywhere."""
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(3), size=(32, 32))
    pred, adjusted = adjusted_prediction(probs, 1e6)
    np.testing.assert_array_equal(pred, np.zeros((32, 32), dtype=np.int64))
    np.testing.assert_allclose(adjusted[..., 0], 1.0, rtol=1e-12)


def test_bias_sweep_proper_scores_match_one_shot() -> None:
    """Accumulated proper scores under a bias match a one-shot reference."""
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(3), size=(8, 8)).astype(np.float32)
    target = rng.integers(0, 3, size=(8, 8))
    delta = 0.5

    accumulators = BiasGridAccumulators((0.0, delta), num_classes=3)
    # Feed the image as four 4x4 tiles through the public accumulator API.
    accumulators.begin_image()
    for r0, r1 in ((0, 4), (4, 8)):
        for c0, c1 in ((0, 4), (4, 8)):
            tile_probs = probs[r0:r1, c0:c1]
            tile_target = target[r0:r1, c0:c1]
            _, adjusted = adjusted_prediction(tile_probs, delta)
            pred = np.argmax(adjusted, axis=-1)
            accumulators.add_tile(
                delta_index=1,
                groups=("overall", "4x"),
                prediction=pred,
                target=tile_target,
                probs_for_scores=adjusted,
            )
    accumulators.end_image(("overall", "4x"))

    final = accumulators.finalize(n_images={"overall": 1, "4x": 1})
    rows = [row for row in final if row[0] == repr(delta) and row[1] == "overall"]
    assert len(rows) == 5  # three classes + object + all

    # One-shot reference computed directly from the full-image adjusted probs.
    _, adjusted_full = adjusted_prediction(probs, delta)
    reference = proper_score_partial_sums(adjusted_full, target, 3)

    all_row = [row for row in rows if row[2] == "all"][0]
    assert float(all_row[7]) == pytest.approx(
        float(reference["brier_sum"] / reference["n_pixels"]), rel=1e-12
    )
    assert float(all_row[8]) == pytest.approx(
        float(reference["log_loss_sum"] / reference["n_pixels"]), rel=1e-12
    )

    # The accumulator counts are exact integers regardless of tiling.
    confusion = accumulators.confusion(1, "overall")
    assert int(confusion.sum()) == 64
    expected = np.zeros((3, 3), dtype=np.int64)
    pred_full = np.argmax(adjusted_full, axis=-1)
    for t in range(3):
        for p in range(3):
            expected[t, p] = int(np.count_nonzero((target == t) & (pred_full == p)))
    np.testing.assert_array_equal(confusion.astype(np.int64), expected)


def test_bias_sweep_bias_zero_matches_argmax_of_raw_probs() -> None:
    """At bias 0 the accumulator confusion equals argmax(raw probs) vs target."""
    rng = np.random.default_rng(4)
    probs = rng.dirichlet(np.ones(3), size=(16, 16)).astype(np.float32)
    target = rng.integers(0, 3, size=(16, 16))
    accumulators = BiasGridAccumulators((0.0,), num_classes=3)
    pred = np.argmax(probs, axis=-1)
    accumulators.begin_image()
    accumulators.add_tile(
        delta_index=0,
        groups=("overall",),
        prediction=pred,
        target=target,
        probs_for_scores=probs,
    )
    accumulators.end_image(("overall",))
    confusion = accumulators.confusion(0, "overall").astype(np.int64)
    expected = np.zeros((3, 3), dtype=np.int64)
    for t in range(3):
        for p in range(3):
            expected[t, p] = int(np.count_nonzero((target == t) & (pred == p)))
    np.testing.assert_array_equal(confusion, expected)


def test_finalize_dice_matches_eval_reduction_on_large_counts() -> None:
    """Pooled Dice must equal eval's reported reduction bit for bit.

    Regression test: with counts above the float32 mantissa, the operation
    order of the two mathematically identical Dice formulas rounds
    differently in float32; the sweep must use the same reduction as the
    eval report (``class_metrics_from_confusion``), not the per-image helper.
    """
    from spheroid_seg.eval import class_metrics_from_confusion

    real_scale = np.array(
        [
            [730_594_794, 933_000, 10_582_889],
            [148_591, 677_788, 983_877],
            [369_263, 680_381, 5_831_977],
        ],
        dtype=np.uint32,
    )
    accumulators = BiasGridAccumulators((0.0,), num_classes=3)
    accumulators.begin_image()
    accumulators.add_tile(
        delta_index=0,
        groups=("overall",),
        prediction=np.zeros((2, 2), dtype=np.int64),
        target=np.zeros((2, 2), dtype=np.int64),
        probs_for_scores=np.full((2, 2, 3), 1.0 / 3.0),
    )
    accumulators.end_image(("overall",))
    accumulators._confusions[0]["overall"] = real_scale  # exact-count injection

    rows = accumulators.finalize(n_images={"overall": 1})
    expected = class_metrics_from_confusion(real_scale)
    for cls_idx, cls in enumerate(("background", "loose cell", "aggregate")):
        (row,) = [r for r in rows if r[2] == cls]
        assert row[3] == float(expected["dice"][cls_idx])
        assert row[4] == float(expected["iou"][cls_idx])


def test_object_precision_recall_hand_computed() -> None:
    """Object metrics derive from the 3x3 matrix with documented semantics."""
    confusion = np.array([[100, 5, 5], [3, 20, 7], [2, 8, 30]], dtype=np.uint32)
    precision, recall = object_precision_recall_from_3x3(confusion)
    # Virtual 2x2: bg_bg=100, bg_obj=10, obj_bg=5, obj_obj=65.
    assert precision == pytest.approx(65 / 75)
    assert recall == pytest.approx(65 / 70)

    per_class = precision_recall_from_confusion(confusion)
    # loose cell: TP=20, colsum=33, rowsum=30.
    assert per_class["precision"][1] == pytest.approx(20 / 33)
    assert per_class["recall"][1] == pytest.approx(20 / 30)


def test_precision_recall_nan_on_zero_denominator() -> None:
    """Precision/recall are NaN when the denominator is zero (never invented)."""
    confusion = np.array([[10, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint32)
    per_class = precision_recall_from_confusion(confusion)
    assert math.isnan(float(per_class["precision"][1]))
    assert math.isnan(float(per_class["recall"][1]))
    # background has predictions and ground truth: well-defined.
    assert per_class["precision"][0] == pytest.approx(1.0)


class _StubPredict:
    """Predict callable returning precomputed probability batches in order."""

    def __init__(self, batches: list[np.ndarray]):
        self._batches = list(batches)
        self.calls = 0

    def __call__(self, params, batch_stats, batch) -> np.ndarray:
        del params, batch_stats, batch
        self.calls += 1
        return self._batches.pop(0)


def test_sweep_full_image_excludes_reflect_padding() -> None:
    """Reflect-padded pixels are excluded from confusion counts and proper scores.

    The 20x12 image is reflect-padded to 32x32 for tiling at tile size 16.
    The padded corners predict a strong foreground class; if padding leaked
    into the accumulators the totals would exceed the 240 real pixels.
    """
    rng = np.random.default_rng(5)
    tile_size = 16
    image_h, image_w = 20, 12
    image = rng.random((image_h, image_w)).astype(np.float32)
    target = rng.integers(0, 3, size=(image_h, image_w)).astype(np.uint8)

    tiles, padding = _extract_tiles_local(image, tile_size)
    assert padding["pad_top"] > 0 or padding["pad_left"] > 0  # padding really exists

    n_tiles = tiles.shape[0]
    probs = np.empty((n_tiles, tile_size, tile_size, 3), dtype=np.float32)
    valid = np.zeros((tile_size, tile_size), dtype=bool)
    slices = _tile_valid_slices(padding)
    for r0, r1, c0, c1 in slices:
        valid[r0:r1, c0:c1] = True
    probs[:] = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    # Strong foreground signal everywhere the tile is reflect padding.
    for t in range(n_tiles):
        padded_view = np.broadcast_to(np.array([0.01, 0.9, 0.09], np.float32), probs[t].shape)
        probs[t] = np.where(valid[..., None], probs[t], padded_view)

    # Single batch covering all four tiles.
    predict = _StubPredict([probs])
    grid = (0.0, 2.0)
    accumulators = BiasGridAccumulators(grid, num_classes=3)
    baseline_pred = sweep_full_image(
        predict_fn=predict,
        params=None,
        batch_stats=None,
        image=image,
        mask=target,
        tile_size=tile_size,
        batch_size=n_tiles,
        num_classes=3,
        accumulators=accumulators,
        mag="4x",
    )

    assert baseline_pred.shape == (image_h, image_w)
    # Baseline argmax of (0.5, 0.3, 0.2) is background everywhere valid.
    np.testing.assert_array_equal(baseline_pred, np.zeros((image_h, image_w), np.uint8))

    gt_counts = np.bincount(target.ravel(), minlength=3)
    for delta_index, delta in enumerate(grid):
        confusion = accumulators.confusion(delta_index, "overall").astype(np.int64)
        assert int(confusion.sum()) == image_h * image_w, f"bias {delta} counted padding"
        np.testing.assert_array_equal(confusion.sum(axis=1), gt_counts)
        if delta > 0:
            # The strong background bias removes every predicted foreground pixel.
            predicted_fg = confusion[:, 1:].sum()
            assert int(predicted_fg) == 0

    # Proper scores at bias 0 match the closed form of the uniform valid probs.
    # The reference uses the float32-cast values: the sweep accumulates the
    # model's float32 probabilities, whose nearest float64 values are what
    # proper_score_partial_sums actually sums.
    final = accumulators.finalize(n_images={"overall": 1, "4x": 1})
    all_row = [row for row in final if row[0] == "0.0" and row[1] == "overall" and row[2] == "all"][
        0
    ]
    p = np.array([0.5, 0.3, 0.2], dtype=np.float32).astype(np.float64)
    # Mean over pixels of sum_c (p_c - y_c)^2 for the uniform valid probs.
    one_hot = np.eye(3)[target]
    expected_brier = float(np.mean(((p - one_hot) ** 2).sum(axis=-1)))
    assert float(all_row[7]) == pytest.approx(expected_brier, rel=1e-12)
    expected_log_loss = float(np.mean(-np.log(p[target.ravel()])))
    assert float(all_row[8]) == pytest.approx(expected_log_loss, rel=1e-12)


def _extract_tiles_local(image: np.ndarray, tile_size: int):
    from spheroid_seg.data.tiling import extract_tiles

    tiles_2d, padding = extract_tiles(image, tile_size)
    return tiles_2d[..., None], padding


def test_analysis_masks_cover_object_and_class_specific() -> None:
    """The three documented analysis masks are exactly the supported set."""
    assert ANALYSIS_MASKS == ("object", "loose cell", "aggregate")


class _FixedDatetime:
    """Frozen datetime so output-directory name collisions are deterministic."""

    _real = datetime.datetime

    @classmethod
    def now(cls, tz=None):
        return cls._real(2026, 10, 1, 12, 0, 0, tzinfo=datetime.UTC)


def test_output_dir_never_overwrites_non_empty(tmp_path, monkeypatch) -> None:
    """A non-empty existing directory is never overwritten; empty ones are reused."""
    import spheroid_seg.validation_diagnostics as vd

    monkeypatch.setattr(vd.datetime, "datetime", _FixedDatetime)
    root = tmp_path / "diagnostics"

    first = vd._make_output_dir(root, "cfg")
    first.joinpath("patch_prevalence.csv").write_text("data")
    second = vd._make_output_dir(root, "cfg")
    assert second != first
    assert second.name == f"{first.name}_1"
    assert second.exists()
    assert not any(second.iterdir())
    # The pre-existing directory and its contents are untouched.
    assert (first / "patch_prevalence.csv").read_text() == "data"

    # An existing but empty directory is reused rather than suffixed.
    reused = root / "cfg_empty_20261001_120000"
    reused.mkdir(parents=True)
    assert vd._make_output_dir(root, "cfg_empty") == reused
