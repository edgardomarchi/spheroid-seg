# Evaluation pipeline

How a trained model is evaluated on a held-out split, from checkpoint to metric
report and overlay grid. For architecture rationale see `docs/design.md`; for
training see `docs/training.md`.

## Overview

```
configs/*.yaml ──► eval.py ──► resolve checkpoint
                      │
                      ├─ dataset.py   real pairs from data/splits/<split>.txt
                      ├─ synthetic.py  deterministic fallback when no real data
                      ├─ tiling.py    full-image tile / predict / reassemble
                      ├─ metrics.py   per-class Dice / IoU + confusion matrix
                      ├─ metadata.py  magnification parsing (filename / CSV)
                      └─ overlays.py  OpenCV grid: raw | GT | pred | errors
                      │
                      ▼
        outputs/evals/<config>_<timestamp>/
            ├── metrics.json            nested report (overall + per mag)
            ├── metrics.csv             flat table (group x class)
            ├── confusion_matrix.csv    global pixel-level matrix
            └── overlays_grid.png       stratified sample grid
```

## Components

- **`src/spheroid_seg/eval.py`** — the evaluation CLI:
  - loads a YAML config; no hardcoded hyperparameters;
  - resolves the checkpoint using `--checkpoint` > `--run-dir` > latest
    `outputs/runs/<config>_*/checkpoints/best_checkpoint.msgpack`;
  - builds the U-Net from the config and restores `params` + `batch_stats`;
  - uses real data when `data/raw/` and `data/masks/` exist, otherwise falls
    back to the deterministic synthetic generator;
  - restricts evaluation to `data/splits/<split>.txt` for real data, or to the
    shared synthetic train/val/test assignment for synthetic data;
  - runs inference in `train=False` mode (BatchNorm running statistics are
    restored, never updated);
  - tiles full images, predicts each tile, and reassembles the full mask;
  - accumulates a global pixel-level confusion matrix and reports pooled and
    per-image Dice/IoU overall and per magnification group;
  - writes all outputs to a unique `outputs/evals/<config>_<timestamp>/`
    directory.

- **`src/spheroid_seg/data/tiling.py`** — non-overlapping square tiling and
  reassembly. Images that are not divisible by the patch size are reflect-padded
  before tiling and cropped back after reassembly, so the round-trip is an
  identity.

- **`src/spheroid_seg/data/metadata.py`** — magnification metadata helpers:
  - `parse_magnification(name)` returns `"4x"`, `"10x"`, or `"unknown"` from the
    filename suffix (the token immediately before the extension must be exactly
    `_4x` or `_10x`);
  - optional `data/metadata.csv` overrides the filename suffix via the precedence
    CSV > filename > `"unknown"`. Malformed CSV rows raise a clear error.

- **`src/spheroid_seg/overlays.py`** — OpenCV overlay grid:
  - rows = up to `eval.num_overlay_samples` images, selected deterministically
    and stratified across the magnification groups present;
  - columns = raw (grayscale) | ground truth | prediction | error overlay;
  - fixed class colormap: background black, loose cell green, aggregate yellow;
  - error overlay: true positives in the class color, false positives in red,
    false negatives in blue, drawn over the raw image with a small legend.

- **`src/spheroid_seg/data/synthetic.py`** — synthetic fallback now names files
  `synth_{idx:03d}_4x.png` / `synth_{idx:03d}_10x.png` (even indices 4x, odd
  10x). 10x objects are drawn with larger radii than 4x objects, while keeping
  the same intensity semantics so the task remains trivially learnable. The
  same deterministic 70/15/15 split is shared between `train.py` and `eval.py`.

## Usage

```bash
# Evaluate the latest run for a config on the val split (default)
uv run python -m spheroid_seg.eval --config configs/base.yaml

# Evaluate the test split
uv run python -m spheroid_seg.eval --config configs/base.yaml --split test

# Use a specific run directory or checkpoint
uv run python -m spheroid_seg.eval --config configs/base.yaml --run-dir outputs/runs/base_20260101_000000
uv run python -m spheroid_seg.eval --config configs/base.yaml --checkpoint outputs/runs/base_20260101_000000/checkpoints/best_checkpoint.msgpack
```

Invalid `--split` values are rejected with a non-zero exit. If no checkpoint can
be resolved, the CLI exits non-zero with a clear message.

## Configurations

The `eval:` section in each config controls evaluation behavior:

| Key | Default in base.yaml | Purpose |
|---|---|---|
| `eval.batch_size` | 4 | Batch size for tiling inference |
| `eval.num_overlay_samples` | 8 | Maximum rows in the overlay grid |
| `eval.overlay_panel_width` | 384 | Pixel size of each square grid panel |

## Determinism

Given the same config and checkpoint, two identical eval invocations produce
identical `metrics.json` files and pixel-identical overlay grids. The output
directory uses a timestamp, so multiple runs never overwrite each other.

## Sanity checks and what to expect on synthetic data

- **Synthetic task is trivially learnable**: after a short CPU training
  (`configs/tiny.yaml`, ~20 epochs), pooled per-class Dice should be >= 0.9
  both overall and within each magnification group (`4x` and `10x`), on val
  and test.
- **Design-doc §8 targets** (aggregate >= 0.85, cell >= 0.75 on **real** data)
  are out of scope for the M4 acceptance; the synthetic acceptance threshold is
  0.9 to leave margin below the >0.98 best-checkpoint result.
- **Unknown magnification** never crashes evaluation; such images are reported
  as their own `"unknown"` group.

## Known limitations

- Tiling is currently non-overlapping; overlapping patches with logit averaging
  will be added in M5 (full-image stitching inference).
- Physical-size normalization is not performed; all sizes are reported in pixels
  because the images' scale bars are unreliable.
