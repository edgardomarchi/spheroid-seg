# Project status

Living snapshot of progress, decisions made after `docs/design.md`, and pending
items. Update at the end of every module. Design rationale lives in
`docs/design.md`; conventions in `AGENTS.md`; this file only tracks *where we are*.

Last updated: 2026-08-14 (first real-data baseline recorded; float32 accumulation bug fixed).

## Modules

| Module | Status | Notes |
|---|---|---|
| M0 — Repo scaffolding | Done | uv + pyproject, src layout, ruff/pytest wired |
| M1 — Data pipeline | Done | dataset, patching (mask-guided), augmentation, synthetic fixtures; visual acceptance re-confirmed on 4 real annotated images; see `docs/data-pipeline.md` |
| M2 — U-Net (Flax) | Done | from scratch, 7.7M params at base_features=32; BatchNorm `batch_stats` verified |
| M3 — Training loop | Done | losses (Dice + weighted CE), metrics, TrainState + AdamW, unique run dirs, `--overfit-one-batch`; `bn_momentum` config param added/fixed (0.9 in base/colab) resolving val divergence on small data; see `docs/training.md`. 48 tests green, ruff clean |
| M4 — Evaluation CLI | Done | per-class Dice/IoU, confusion, overlays, per-magnification grouping, checkpoint resolution; pooled confusion matrix now uses exact uint32 accumulation (fixed float32 saturation at 2**24); regression test added; see `docs/evaluation.md` |
| M5 — Inference (stitching) | Done | full-image patch stitching with logit averaging; see `docs/inference.md` |
| Notebook — Colab quickstart | Done | `notebooks/colab_training.ipynb` converted to pip-based install (`pip install -e ".[cuda12,viz]"` GPU / `pip install -e ".[viz]"` CPU), `USE_DRIVE_DATA` flag, streaming `run()` helper, pytest cell removed; `configs/colab.yaml` added for T4 16 GiB (batch_size 4); see `docs/training.md` |
| CI — GitHub Actions | Done | lint + `pytest` matrix on Python 3.12/3.13/3.14; see `.github/workflows/ci.yml` |
| M6 — Post-processing (v0.2) | Pending | instances, morphometrics, hybrid spheroid/organoid classification |
| M7 — SLiMIA pre-training | Deferred | domain-shift risk; only if own data underperforms |

## First real-data baseline

Dataset: 21 real annotated phase-contrast images (4x and 10x), masks with IDs 0–3
per spec; the model trains 3 classes (background / loose cell / aggregate, IDs 2+3
merged). The first real-data training run was `outputs/runs/colab_20260812_182023`
(`configs/base.yaml`, `bn_momentum: 0.9`, 100 max epochs, early-stopping patience
20 → stopped at epoch 47, best checkpoint at ~epoch 27).

Baseline computed on the validation split — the same split that guided early stopping, so these numbers are mildly optimistic. The test split remains untouched and is reserved for the final v0.1 acceptance check (§8 criteria).

**Caveat:** the validation split used for this baseline has only **3 images** (2× 10x,
1× 4x). All numbers below are noisy and must be re-evaluated once the full
clinical batch (~80–100 images) arrives.

### Post-fix validation Dice (background / loose cell / aggregate)

| Group | n | background | loose cell | aggregate |
|---|---|---:|---:|---:|
| overall | 3 | 0.9907 | 0.3554 | 0.2954 |
| 10x | 2 | 0.9888 | 0.3505 | 0.2983 |
| 4x | 1 | 0.9945 | 0.3664 | 0.2856 |

Per-image Dice:

| Image | mag | background | loose cell | aggregate |
|---|---|---:|---:|---:|
| 162 - 1d 3T3_10x | 10x | 0.9901 | 0.3756 | 0.3773 |
| 49 - 1d 3T3_10x | 10x | 0.9875 | 0.2859 | 0.1862 |
| 219 - 4d 3T3_4x | 4x | 0.9945 | 0.3664 | 0.2856 |

### Confusion matrix on validation (rows = GT, columns = prediction)

Verified: row sums equal the actual mask pixel counts; total = 3 × 6128 × 8168 =
150,160,512 px.

| GT \ pred | background | loose cell | aggregate |
|---|---:|---:|---:|
| background | 145,127,942 | 351,148 | 2,236,410 |
| loose cell | 70,879 | 434,604 | 426,133 |
| aggregate | 61,393 | 728,261 | 723,742 |

Headline: background is clean (~1.5% false positive); loose↔aggregate
cross-confusion is ~50% in both directions and dominates the error.

### GT audit conclusion

Connected-components check on 3 representative masks found **zero mixed-class
objects** and **0% of pixels within 20 px of the other foreground class**. The
annotations are internally consistent; the loose↔aggregate confusion is a genuine
semantic-discrimination limitation, not annotation noise or a pipeline bug.

### Eval-metric bug found and fixed at baseline time

Pooled metrics were corrupted by float32 accumulation saturating at exactly 2**24
(background→background cell froze at 16,777,216). Fixed by exact uint32
accumulation in `eval.py::accumulate_confusion_matrix` and
`metrics.py::_per_class_counts`, with a regression test accumulating > 2**24
pixels of one class. All numbers above are post-fix (verified run:
`outputs/evals/base_20260814_132521/`).

### Qualitative overlay observations

- Speckle false positives across the field, especially visible at 4x.
- Systematic false positives along the dark well rim (border artifact predicted
  as object).
- False-negative patches inside objects match the loose↔aggregate confusion seen
  in the matrix.

## Decisions made after the design doc

- **Synthetic fixtures must correlate raw intensities with masks** (M3): pure
  noise made generalization impossible. Background dark, class 1 medium, class 2
  bright; large aggregates drawn before small loose cells so both classes keep
  visible area.
- **Unique run directories** (`outputs/runs/<config>_<timestamp>/`) after an
  orphaned training process collision; rule added to `AGENTS.md`: one training at
  a time, check `pgrep -f spheroid_seg.train` before launching.
- **Hardware strategy** (design doc §5): CPU-first dev, free cloud GPU tier
  (Colab) as the primary full-training path, local ROCm iGPU pending JAX
  verification.
- **Determinism is a hard requirement**, verified manually: identical commands
  must give identical losses. Re-verify after any pipeline change.
- **Synthetic train/val/test split is shared** between `train.py` and `eval.py`
  via `synthetic_split_names` so evaluation never sees training images.
- **Synthetic filenames carry magnification** (`_4x` / `_10x`) and 10x objects
  are drawn larger than 4x objects, making the per-magnification breakdown
  meaningful while keeping the task learnable.
- **ROCm iGPU experiment outcome** (M5): `jax.devices()` only reports the CPU on
  the test machine's iGPU (gfx1152) under ROCm 7. Local development stays
  CPU-first; the GPU path remains Colab/cloud CUDA, as stated in the design doc.
- **Colab GPU notebook** (cloud workflow): `notebooks/colab_training.ipynb`
  added to implement design doc §5 (clone + uv sync + GPU check + pytest +
  `base.yaml --overfit-one-batch` + checkpoint download). The notebook drives
  all commands through subprocesses so the Colab kernel stays stock. Actual GPU
  execution is a manual maintainer verification step.
- **Real-data onboarding started** (2026-08-06): 4 annotated images (3× 10x,
  1× 4x) exported from QuPath via `scripts/export_qupath_masks.groovy`, QC
  passed, splits committed. Annotation pitfalls documented in
  `docs/data-pipeline.md` (area tools only, PathClass required).
- **Filename convention for real data**: base names end in `_4x`/`_10x`
  (magnification parser contract, same as synthetic fixtures). To be
  communicated to the clinical group for the full batch.
- **Two bugfixes from the real-data visual check**: `_overlay()` now scales
  grayscale float images to uint8 (image channel was truncated to black);
  scale augmentation uses reflect border mode (no artificial black frames;
  mask reflects consistently). Regression tests added.
- **Split layout with n=4**: train = 2× 10x, val = 1× 10x + 1× 4x — train
  temporarily has no 4x; accepted artifact of a too-small-to-stratify group,
  resolves when the full batch arrives. `test.txt` intentionally empty;
  `eval --split test` fails cleanly ("No images found for split 'test'").
- **D2 evidence (2026-08-14)**: the baseline loose↔aggregate cross-confusion
  dominates the error and the GT audit is clean, so the loose/aggregate
  distinction may belong to v0.2 object-level classification rather than
  pixel-level training. Decision deferred to the ~50-image mark: stay 3-class or
  collapse to 2-class (background / object).
- **D4 evidence (2026-08-14)**: no per-magnification failure signal so far
  (per-image background Dice 10x ≈ 4x), but val n is 2 vs 1 — evidence is weak.
  Single model stays; revisit with more data.

## Pending — v0.1 public release

1. **Await the full clinical batch (~80–100 annotated images).** Ingestion
   workflow once images arrive: QC → `scripts/make_metadata.py` →
   `scripts/make_splits.py --force` → commit new splits → upload to Drive →
   Colab training (`configs/colab.yaml`) → eval → compare against the baseline
   recorded above.
2. **Decide 2-class vs 3-class at ~50 annotated images** (D2 escape hatch).
   Current evidence supports deferring the loose/aggregate distinction to v0.2
   object-level classification, but the decision needs more data.
3. **Add a combined "object" (class 1 ∪ class 2) Dice metric** to eval as a
   secondary output. Recommended; maintainer has not confirmed yet.
4. **Meet v0.1 release criteria** (design doc §8): aggregate Dice ≥ 0.85,
   loose-cell Dice ≥ 0.75 → tag v0.1 when met.
5. **Zenodo sample publication + `scripts/download_data.py` fetch step**
   (blocked on clinical approval).
6. **Decide whether `data/metadata.csv` should be committed** (currently local).

## Pending — technical

- **`train.py` resume-from-checkpoint** (design doc §5): training must be
  resumable because cloud sessions can be cut, but `train.py` only saves
  checkpoints and has no `--resume` path.
- **Speckle false positives (4x) and dark-well-rim false positives** observed in
  baseline overlays: a minimum-object-size filter and possible border handling
  are v0.2 post-processing scope; no action in v0.1.

## Pending — clinical group

- ~80–100 properly annotated images (masks per spec in design doc §2.2),
  expected within a week of project start; ~400 later.
- Objective spheroid/organoid criteria + borderline gallery (needed for v0.2).
- Final metrics/units decision (scale bars are wrong — pixels vs real µm/px
  calibration).
