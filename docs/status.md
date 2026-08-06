# Project status

Living snapshot of progress, decisions made after `docs/design.md`, and pending
items. Update at the end of every module. Design rationale lives in
`docs/design.md`; conventions in `AGENTS.md`; this file only tracks *where we are*.

Last updated: 2026-08-06 (real-data onboarding started).

## Modules

| Module | Status | Notes |
|---|---|---|
| M0 — Repo scaffolding | Done | uv + pyproject, src layout, ruff/pytest wired |
| M1 — Data pipeline | Done | dataset, patching (mask-guided), augmentation, synthetic fixtures; visual acceptance re-confirmed on 4 real annotated images; see `docs/data-pipeline.md` |
| M2 — U-Net (Flax) | Done | from scratch, 7.7M params at base_features=32; BatchNorm `batch_stats` verified |
| M3 — Training loop | Done | losses (Dice + weighted CE), metrics, TrainState + AdamW, unique run dirs, `--overfit-one-batch`; see `docs/training.md`. 48 tests green, ruff clean |
| M4 — Evaluation CLI | Done | per-class Dice/IoU, confusion, overlays, per-magnification grouping, checkpoint resolution; see `docs/evaluation.md` |
| M5 — Inference (stitching) | Done | full-image patch stitching with logit averaging; see `docs/inference.md` |
| M6 — Post-processing (v0.2) | Pending | instances, morphometrics, hybrid spheroid/organoid classification |
| M7 — SLiMIA pre-training | Deferred | domain-shift risk; only if own data underperforms |

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

## Pending — v0.1 public release

- Real annotated data: first batch of masks per spec in `docs/design.md` §2.2 and
  the first real train/val leak check.
- Full `configs/base.yaml --overfit-one-batch` acceptance check on GPU/Colab
  (impractical on CPU; smoke-config substitute passed).
- CI pipeline (GitHub Actions) covering Python 3.12/3.13/3.14.
- Zenodo sample publication + `scripts/download_data.py` fetch step.
- Real annotated data: first 4 images wired and validated; awaiting the full
  clinical batch (~80–100) and the first real train/val leak check.

## Pending — technical

- Run the full `configs/base.yaml --overfit-one-batch` acceptance check on
  GPU/Colab (impractical on CPU; smoke-config substitute passed).
- Real-data wiring: the stratified split script (`scripts/make_splits.py`) is
  done; what remains is the arrival of the first batch of annotated images and
  the first real train/val leak check.
- Optional: `--seed` CLI override for `visualize_batches.py`.
- Real-data wiring: done for the first 4 images (QC + splits + visual check).
  Remaining: full-batch ingestion via `scripts/make_splits.py` and the first
  real train/val leak check when the clinical batch arrives.

## Pending — clinical group

- ~80–100 properly annotated images (masks per spec in design doc §2.2),
  expected within a week of project start; ~400 later.
- Objective spheroid/organoid criteria + borderline gallery (needed for v0.2).
- Final metrics/units decision (scale bars are wrong — pixels vs real µm/px
  calibration).
