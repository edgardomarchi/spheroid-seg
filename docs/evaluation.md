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
                      ├─ metrics.py   per-class Dice / IoU + confusion matrix + proper scoring rules
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

## Memory behavior

Evaluation is streaming by design: it loads one image, predicts, folds the
result into the pooled confusion counts and per-image scalar metrics, then
releases the image. Only the small aggregated state and the few overlay panels
selected for the grid are retained, so peak RSS stays bounded by one image
plus the overlay sample budget regardless of split size.
`tests/test_eval_memory.py` guards this property: it asserts that an 8-image
eval run stays below 2x the peak RSS of a 2-image run.

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

## Combined "object" metric

In addition to the three model classes, eval reports a secondary **object**
metric that merges the two foreground classes:

- `object = (class == 1) | (class == 2)` — any foreground pixel.
- `background = (class == 0)` — the complement.

The object Dice/IoU is computed from a virtual 2x2 confusion matrix derived
from the standard 3x3 matrix:

```
                 predicted
              bg        object
GT bg    C[0,0]    C[0,1] + C[0,2]
   object C[1,0] + C[2,0]    C[1,1] + C[1,2] + C[2,1] + C[2,2]
```

This metric is decision-relevant for the v0.2 design choice (D2 escape hatch,
see `docs/design.md` §4): the first real-data baseline showed that loose cells
and aggregates are frequently cross-confused at the pixel level while still
being correctly separated from background. The object score measures "how well
do we separate foreground from background?" independent of the harder
loose/aggregate distinction. Only the object row is reported; the virtual
background row is redundant with the existing background class.

The object row appears in `metrics.csv`, `metrics.json`, the stdout summary
table, and a separate `confusion_matrix_object.csv`. The 3x3 confusion matrix
is unchanged.

## Proper scoring rules (Brier score and log loss)

Dice/IoU evaluate the post-argmax mask: they say nothing about the quality or
calibration of the predicted probabilities. This matters here because training
deliberately de-calibrates probabilities (class-weighted cross-entropy with a
low background weight) and because ~98.6% background pixels would dominate any
pooled score unless it is also reported per class. The eval therefore reports
two strictly proper scoring rules (Gneiting & Raftery 2007, JASA 102(477))
computed from the softmax probabilities, never from the argmax mask:

- `brier` — per-class one-vs-rest Brier score: mean over pixels of
  `(p_c − y_c)²`, with `y` the one-hot ground truth. Defined for every class
  (an absent class scores `mean(p_c²)`).
- `log_loss` — per-class conditional log loss: `−mean(log p_true)` restricted
  to pixels whose ground-truth class is `c`; it measures calibration *within*
  each class's GT region. A class absent from the GT has no such pixels: the
  value is NaN (empty cell in `metrics.csv`), never an invented convention.
- The extra `all` row per group carries the pooled multiclass Brier score
  (mean over pixels of `Σ_c (p_c − y_c)²`, range [0, 2]) and the overall log
  loss (`−mean(log p_true)` over all pixels).
- The `object` row leaves both cells empty: object is a post-hoc merge of two
  softmax classes, not a model output, so no probabilities exist for it.

Implementation notes:

- Input convention: probabilities (softmax output), not logits.
- Probabilities are clipped to `[1e-7, 1]` before `log` (fixed deterministic
  constant, not a config key, so scores stay comparable across runs).
- Accumulated as float64 running sums + int64 pixel counts per class, group,
  and overall — per-pixel probability maps are never retained beyond the
  current tile/batch (same streaming contract as the confusion matrix).

Schema: `metrics.csv` gains `brier` and `log_loss` columns alongside
`dice`/`iou`; each group block (overall + per magnification) is the three
class rows, the `object` row, and the `all` row. `metrics.json` gains `brier`,
`brier_all`, `log_loss`, and `log_loss_all` under `overall` and each
`per_magnification` entry.

## Validation diagnostics

`src/spheroid_seg/validation_diagnostics.py` adds a read-only diagnostics
command that explains the current model's foreground overprediction before
any training change is considered. It analyzes a selected split (normally
`val`) in a single streaming pass and never trains, resumes, or touches the
test split:

```bash
uv run python -m spheroid_seg.validation_diagnostics \
    --config configs/base.yaml \
    --run-dir outputs/runs/colab_drive_20260930_120108 \
    --split val \
    --background-bias-grid 0,0.25,0.5,0.75,1,1.5,2 \
    --area-thresholds 16,64,256,1024,4096
```

Options: `--config`, `--split` (`val`/`test`), `--run-dir`/`--checkpoint`
(same resolution as eval), `--background-bias-grid` (comma-separated,
default `0,0.25,0.5,0.75,1,1.5,2`), `--area-thresholds` (comma-separated
pixels, default `16,64,256,1024,4096`), `--output-root` (default
`outputs/diagnostics/`), `--skip-saved-patch-prevalence`, and `--max-images`
(smoke/debug only — never use it for a real-data acceptance run). Every
invocation writes to a unique `outputs/diagnostics/<config>_<timestamp>/`
directory and never overwrites an existing non-empty one.

### Part 1 — class prevalence

Exact class-pixel counts (uint64) for the full-image train/val masks listed
in `data/splits/{train,val}.txt` and for the exact saved augmented patches
in `<run-dir>/checkpoints/training_patches.npz` (the NPZ schema written by
`train.py::save_training_state` is validated first: a missing file or an
uninterpretable schema fails with a clear error — a different patch set is
never silently reconstructed; use `--skip-saved-patch-prevalence` to analyze
full images only, which the summary records). Original mask IDs 2 and 3 are
counted together as model class 2 via the config's `class_mapping`. Full
images are processed one at a time; NPZ arrays one at a time and released.

`patch_prevalence.csv` — long format `source,split,class_name,pixel_count,
pixel_fraction` with `source` ∈ `full_image`/`saved_patch`; `summary.md`
adds foreground/loose/aggregate fractions per source/split and the ratio
between saved-patch and full-image foreground prevalence.

### Part 2 — false-positive connected components

For the baseline bias 0, the reassembled full-image argmax prediction (the
same tiling/reassembly as eval) is decomposed with 8-connectivity
(`scipy.ndimage.label`) into connected components, per analysis mask:

- `object`: predicted foreground (class 1 or 2) where ground truth is
  background;
- `loose cell`: predicted class 1 where ground truth is not class 1;
- `aggregate`: predicted class 2 where ground truth is not class 2.

For every component, `fp_components.csv` records image, magnification,
analysis mask, component ID, area (px), inclusive bounding box, centroid,
minimum distance to the image border, whether it touches the border, and
the mean/median normalized image intensity inside the component.
`fp_component_summary.csv` groups by image/magnification/analysis mask with
total FP area, component count, mean/median/max area, area (and fraction)
below each configured threshold, and border-touching area (and fraction).
Acceptance invariant: component areas sum exactly to the false-positive
pixel count derived from the image's confusion counts.

### Part 3 — background-logit-bias sweep

A post-hoc calibration probe: for a grid of biases `delta`, the background
logit is raised by `delta` and the metrics are recomputed without
retraining. The prediction function exposes softmax probabilities, so the
bias is applied as the mathematically equivalent log-probability shift
`softmax(log p + delta * one_hot(background))`; boosting the background can
only move argmax decisions toward background, so the predicted foreground
count is monotonically non-increasing in `delta`, and a sufficiently large
bias predicts background wherever the float32 background probability did not
underflow to exactly 0 (underflowed pixels follow exact softmax semantics
and can never flip). At `delta == 0` the raw probabilities are used with no
transform.

Each tile is predicted once and every grid accumulator is updated from that
tile. Per bias value and group (`overall` plus magnification groups) the
sweep accumulates the pooled 3x3 confusion matrix (exact uint32) and the
float64 proper-score partial sums, then reports in `bias_sweep.csv` (long
format): `background_bias,group,class,dice,iou,precision,recall,brier,
log_loss,gt_pixels,pred_pixels,n_images` — one row per model class, plus an
`object` row (Dice/IoU/precision/recall derived from the 3x3 matrix with
the semantics of §Combined "object" metric; proper scores empty because the
object class is a post-hoc merge, not a softmax output) and an `all` row
(multiclass Brier + overall log loss). Per-class precision is TP/column
sum, recall TP/row sum, NaN when the denominator is zero. Reflect-padded
tile pixels are excluded from every count and proper score. A companion
`bias_sweep_confusion.csv` (`background_bias,group,gt,prediction,count`)
makes the per-bias confusion matrices directly inspectable.

Acceptance invariants (tested in `tests/test_validation_diagnostics*.py`):

1. bias `0` reproduces the existing eval's pooled confusion matrix exactly;
2. bias `0` reproduces the eval proper scores exactly — proper sums are
   staged per image and folded once per image, the same nesting as the eval
   path (only the summation order tolerance would apply otherwise);
3. predicted foreground pixels are monotonic non-increasing in the bias;
4. a sufficiently large positive bias predicts background everywhere
   (modulo the float32 exact-zero corner documented above);
5. object metrics derive from the 3x3 matrix with the documented
   semantics.

### Memory and streaming guarantees

The diagnostics loop streams like eval: one image is loaded, tiled, and
predicted once; per-tile probabilities feed all bias accumulators and the
baseline reassembly, then are released. Across images only small aggregated
state survives (uint32 confusion counts, float64 proper-score partial sums,
per-component scalar rows) — per-image probability maps are never retained.
Part 1 reads one mask or one NPZ array at a time. Peak RSS therefore stays
bounded by one image plus a constant accumulator budget regardless of split
size; `tests/test_validation_diagnostics_memory.py` guards this the same way
as the eval memory test (8-image peak RSS < 2x the 2-image peak).

## Known limitations

- Tiling is currently non-overlapping; overlapping patches with logit averaging
  will be added in M5 (full-image stitching inference).
- Physical-size normalization is not performed; all sizes are reported in pixels
  because the images' scale bars are unreliable.
