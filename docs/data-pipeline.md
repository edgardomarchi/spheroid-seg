# Data Pipeline

How raw images and annotations become training batches.

## Real data onboarding

When annotated real images arrive:

1. **Run QC**: `uv run python -m spheroid_seg.data.qc --raw-dir data/raw --mask-dir data/masks`.
   Fix any spec violations (wrong dimensions, values outside {0,1,2,3}, etc.).
2. **Generate `data/metadata.csv`**: `uv run python scripts/make_metadata.py --config configs/base.yaml`.
   The script scans `data/raw/`, parses the trailing `_4x` or `_10x` suffix
   (case-insensitive on the extension), and writes the CSV with header
   `image,magnification,condition`. The `condition` column is left empty for
   manual completion. Use `--update` to append newly added images while
   preserving existing rows and their conditions; use `--force` to regenerate
   from scratch.
3. **Fill `condition` by hand** if you want joint stratification by
   magnification and condition. Skip this step for magnification-only splits.
4. **Create the splits**: `uv run python scripts/make_splits.py --config configs/base.yaml`.
   This writes deterministic, stratified `data/splits/{train,val,test}.txt` files.
   Use `--force` only when you intentionally want to overwrite existing splits.
5. **Train/eval as usual**: `uv run python -m spheroid_seg.train --config configs/base.yaml`
   and `uv run python -m spheroid_seg.eval --config configs/base.yaml`.

## Overview

```
data/raw/    (original JPG/TIFF images, read-only)
data/masks/  (annotation PNGs, IDs 0-3, same base name)
     │
     ▼
┌─────────────┐  Spec check: format, matching shape, values in {0,1,2,3}
│   qc.py     │  → outputs/qc/qc_report.csv
└─────────────┘
     │
     ▼
┌─────────────┐  Image-level train/val/test assignment
│  splits.py  │  reads data/splits/*.txt, detects cross-split leakage
└─────────────┘
     │
     ▼
┌─────────────┐  Loads image+mask pairs; per-image percentile 1-99
│ dataset.py  │  normalization to [0,1]; merges classes 2+3 → "aggregate"
└─────────────┘
     │
     ▼
┌─────────────┐  Cuts patch_size x patch_size patches; mask-guided
│ patching.py │  oversampling (~80% object-containing patches)
└─────────────┘
     │
     ▼
┌─────────────┐  flips, 90° rotations, elastic, zoom 0.5-2x, blur, noise,
│ augment.py  │  brightness/contrast jitter — applied jointly to image+mask
└─────────────┘
     │
     ▼
  (image, mask) batches  →  model (M2/M3)
```

## Modules

- **qc.py** — Annotation gatekeeper. CLI:
  `uv run python -m spheroid_seg.data.qc --raw-dir data/raw --mask-dir data/masks`.
  Run this FIRST whenever new annotations arrive. Exits non-zero on spec
  violations; writes a CSV report to `outputs/qc/`.
- **make_metadata.py** — Generates `data/metadata.csv` from `data/raw/`
  filenames. Recognizes `_4x` and `_10x` suffixes at the end of the stem
  (extension case is ignored) and leaves `condition` empty for manual fill.
  Files without a recognized suffix are kept with an empty `magnification`;
  the script prints a warning listing them. Use `--update` to merge new images
  without overwriting manual edits, or `--force` to regenerate from scratch.
- **splits.py** — Reads `data/splits/{train,val,test}.txt`. Never re-shuffles;
  raises if the same image appears in two splits (patch-level leakage guard).
- **dataset.py** — Paired loading, intensity normalization, `input_channels`
  (rgb|grayscale), and class remapping via `class_mapping` (spheroid+organoid
  merge into "aggregate" in memory; original PNGs untouched).
- **patching.py** — Object-containing patches are detected from the MASK
  (fraction of pixels with class &gt; 0 above `min_object_fraction`), never from
  image content — background artifacts become "hard negative" background
  patches. Controlled by `min_object_fraction` and `object_patch_ratio`.
- **augment.py** — Albumentations pipeline applied identically to image and
  mask. The zoom 0.5-2x doubles as the scale-invariance strategy for the two
  magnifications (4x/10x), since physical scale bars are unreliable.
- **scripts/visualize_batches.py** — Visual sanity check: saves an 8-sample
  augmented grid to `outputs/debug/augmented_batches.png` (synthetic fixtures
  when no real data). Run it after any config change to augmentation/patching.

## Key properties

- Everything is driven by `configs/base.yaml`; no hardcoded hyperparameters.
- Deterministic given `seed`: same seed → same patch sequence and same debug
  grid (reproducible debugging).
- The pipeline is framework-agnostic up to the batch: arrays come out as
  NumPy; JAX conversion happens in the training loop (M3).
  
## Annotation export (QuPath)

`scripts/export_qupath_masks.groovy` rasterizes QuPath annotations into the
mask PNGs expected in `data/masks/` (IDs: 0=background, 1=loose cell,
2=spheroid, 3=organoid). Usage: open the image(s) in a QuPath project,
paste the script into Automate → Script editor, Run (or Run for project).
Class names must match exactly: `Loose cell`, `Spheroid`, `Organoid`.
Notes:
- Use area tools (ellipse/polygon/brush), never the points tool — lines
  have no area and rasterize as near-zero pixels.
- Every annotation must have its PathClass assigned (right-click → Set class).
- Exported PNGs may be palette-based; convert to plain grayscale before QC.
- Validate with qc.py before copying into data/masks/.
