# Data Pipeline

How raw images and annotations become training batches.

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