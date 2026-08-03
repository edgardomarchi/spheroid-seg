# Inference pipeline

How a trained model produces full-size predicted masks on new raw images, using
overlapping-patch stitching with logit averaging. For training see
`docs/training.md`; for evaluation see `docs/evaluation.md`.

## Overview

```
configs/*.yaml ──► infer.py ──► resolve checkpoint
                      │
                      ├─ dataset.py   percentile normalization, grayscale/RGB
                      ├─ stitching.py overlapping tiles + logit averaging
                      ├─ models/unet.py Flax U-Net (train=False, batch_stats frozen)
                      └─ overlays.py  quick-look prediction overlay
                      │
                      ▼
        outputs/infer/<config>_<timestamp>/
            ├── predictions/      full-size uint8 masks (IDs 0-2)
            └── overlays/         raw + prediction quick-look PNGs
```

## Stitching

Full microscopy images are usually larger than the model's training patch size,
so inference cuts a regular grid of overlapping square tiles, predicts each
tile independently, and stitches the results back together.

Key design points (frozen for v0.1):

- **Overlap is configurable** via `infer.overlap` (default `0.15`). Stride is
  derived as `round(patch_size * (1 - overlap))`.
- **Reflect padding** ensures the grid covers the whole image. Images smaller
  than one patch are padded up to `patch_size`.
- **Logit averaging**: each tile contributes its per-pixel logits to a float32
  accumulator; a count canvas tracks how many tiles cover each pixel. The
  averaged logits are argmaxed once to produce the final mask. No per-patch
  voting or windowed weighting in v0.1.
- **BatchNorm is frozen**: the forward pass uses `train=False`, so running
  statistics from the checkpoint are used and never updated.

## CLI usage

```bash
# Use the latest run for a config
uv run python -m spheroid_seg.infer --config configs/base.yaml --input data/raw/

# Override checkpoint resolution
uv run python -m spheroid_seg.infer --config configs/base.yaml --input data/raw/ \
    --run-dir outputs/runs/base_20260101_000000
uv run python -m spheroid_seg.infer --config configs/base.yaml --input image_001_4x.png \
    --checkpoint outputs/runs/base_20260101_000000/checkpoints/best_checkpoint.msgpack
```

`--input` may be a single image file or a directory of images. Directory scans
are non-recursive; unsupported extensions are skipped with a warning. The CLI
exits non-zero for a missing input path or an unresolvable checkpoint.

## Configuration

The `infer:` section in each config controls stitching behavior:

| Key | Default in base.yaml | Purpose |
|---|---|---|
| `infer.overlap` | `0.15` | Fractional overlap between adjacent tiles |
| `infer.batch_size` | `4` | Batch size for tile inference |

## Outputs

- `predictions/<image_stem>.png`: uint8 grayscale mask with class IDs 0-2,
  identical pixel size to the input image.
- `overlays/<image_stem>.png`: quick-look overlay of the predicted classes over
  the normalized raw image.

A short summary is printed to stdout:

```
Inference summary
  Images processed: 4
  Output directory: outputs/infer/base_20260803_123456
```

## CPU expectations

Like the rest of the pipeline, inference is CPU-first. `configs/tiny.yaml` and
`configs/smoke.yaml` run comfortably on a CPU-only machine. The synthetic
acceptance check (one 4x and one 10x image) produces complete masks with
per-class Dice well above 0.9 after a short `tiny.yaml` training.
