# Project status

Living snapshot of progress, decisions made after `docs/design.md`, and pending
items. Update at the end of every module. Design rationale lives in
`docs/design.md`; conventions in `AGENTS.md`; this file only tracks *where we are*.

Last updated: 2026-07-31 (after M3).

## Modules

| Module | Status | Notes |
|---|---|---|
| M0 — Repo scaffolding | Done | uv + pyproject, src layout, ruff/pytest wired |
| M1 — Data pipeline | Done | dataset, patching (mask-guided), augmentation, synthetic fixtures; see `docs/data-pipeline.md` |
| M2 — U-Net (Flax) | Done | from scratch, 7.7M params at base_features=32; BatchNorm `batch_stats` verified |
| M3 — Training loop | Done | losses (Dice + weighted CE), metrics, TrainState + AdamW, unique run dirs, `--overfit-one-batch`; see `docs/training.md`. 48 tests green, ruff clean |
| M4 — Evaluation CLI | Next | per-class Dice/IoU, confusion, overlays; must report metrics per magnification |
| M5 — Inference (stitching) | Pending | full-image patch stitching |
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

## Pending — technical

- Run the full `configs/base.yaml --overfit-one-batch` acceptance check on
  GPU/Colab (impractical on CPU; smoke-config substitute passed).
- Verify whether JAX sees the notebook iGPU under ROCm (`jax.devices()`).
- Real-data wiring: `data/splits/*.txt` + train/val leak check when annotated
  images arrive (stratified split script task).
- Optional: `--seed` CLI override for `visualize_batches.py`.

## Pending — clinical group

- ~80–100 properly annotated images (masks per spec in design doc §2.2),
  expected within a week of project start; ~400 later.
- Objective spheroid/organoid criteria + borderline gallery (needed for v0.2).
- Final metrics/units decision (scale bars are wrong — pixels vs real µm/px
  calibration).
