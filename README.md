# spheroid-seg

Open-source pipeline for segmenting spheroids and loose cells in phase-contrast
microscopy images (4x and 10x). Version 0.1 trains a single U-Net in JAX/Flax
for 3-class semantic segmentation:

- **0 — background**
- **1 — loose cell**
- **2 — aggregate** (spheroid and organoid are merged at training time)

> **Project status:** the pipeline is complete and runs end-to-end, but it is
> currently validated only on synthetic smoke-test data. The first real annotated
> baseline is pending. The design document sets orientation targets of aggregate
> Dice ≥ 0.85 and loose-cell Dice ≥ 0.75 on real data; these are targets, not
> achieved results.

## Scope

| Version | What is in this repo | What is planned |
|---|---|---|
| **v0.1 (current)** | Semantic segmentation: data pipeline, Flax U-Net, training loop, evaluation CLI, and full-image inference via overlapping-patch stitching | — |
| **v0.2 (planned)** | Instance separation (watershed), morphometrics, and hybrid spheroid / organoid classification | See `docs/design.md` §3 |

## Installation

You need Python `>=3.12,<3.15` and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
```

This installs JAX/Flax, Optax, Albumentations, OpenCV, scikit-image, and the
dev tools (pytest, ruff).

## Quickstart (no data required)

The repo works out of the box with synthetic fixtures, so you can verify the
full pipeline on a CPU-only laptop in a few minutes.

```bash
# 1. Run the test suite
uv run pytest

# 2. Visualize a batch of augmented patches
uv run python scripts/visualize_batches.py --config configs/tiny.yaml
#    → outputs/debug/augmented_batches.png

# 3. Train a small model for a few epochs on synthetic data
uv run python -m spheroid_seg.train --config configs/tiny.yaml --epochs 5
#    → outputs/runs/tiny_<timestamp>/

# 4. Evaluate the latest checkpoint on the synthetic test split
uv run python -m spheroid_seg.eval --config configs/tiny.yaml --split test
#    → outputs/evals/tiny_<timestamp>/

# 5. Create a single synthetic input image and run full-image inference
mkdir -p outputs/infer_input
uv run python -c "
import cv2, numpy as np
rng = np.random.default_rng(42)
img = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
cv2.imwrite('outputs/infer_input/synth_4x.png', img)
"
uv run python -m spheroid_seg.infer --config configs/tiny.yaml \
    --input outputs/infer_input/
#    → outputs/infer/tiny_<timestamp>/
```

For a real training run, switch to `configs/base.yaml` (512² patches, 7.7M
parameters); it is intended for a GPU or Colab T4. See `docs/training.md` for
longer runs and the `--overfit-one-batch` sanity check.

## Data expectations and real-data onboarding

The repository does **not** contain images. The expected layout is:

```
data/
  raw/      # original microscopy images (JPG/TIFF)
  masks/    # grayscale PNG annotations, same base name and size as the raw image
  splits/   # train.txt / val.txt / test.txt (image-level, never patch-level)
```

Annotation format:

- Grayscale PNG, uint8, same dimensions as the raw image.
- Pixel values are class IDs:
  - `0` = background
  - `1` = loose cell
  - `2` = spheroid
  - `3` = organoid
- The model trains with 3 classes: IDs `2` and `3` are merged in memory as
  "aggregate" (the original PNGs are left untouched).

When annotated real images arrive:

1. Run QC:
   `uv run python -m spheroid_seg.data.qc --raw-dir data/raw --mask-dir data/masks`
2. Fill `data/metadata.csv` with at least `image,magnification` (optionally add
   `condition` for joint stratification).
3. Create stratified splits:
   `uv run python scripts/make_splits.py --config configs/base.yaml`
4. Train and evaluate as usual:
   `uv run python -m spheroid_seg.train --config configs/base.yaml`
   `uv run python -m spheroid_seg.eval --config configs/base.yaml`

A public sample subset will be published on Zenodo once clinical-group approval
is complete; `scripts/download_data.py` is currently a placeholder for that
future fetch step.

## Documentation map

| File | What it covers |
|---|---|
| [`docs/design.md`](docs/design.md) | Full design spec: goals, data format, architecture decisions, roadmap |
| [`docs/data-pipeline.md`](docs/data-pipeline.md) | Real-data onboarding, QC, patching, augmentation |
| [`docs/training.md`](docs/training.md) | Training loop, configs, determinism, sanity checks |
| [`docs/evaluation.md`](docs/evaluation.md) | Evaluation CLI, metrics, overlay grids |
| [`docs/inference.md`](docs/inference.md) | Full-image stitched inference CLI |
| [`docs/status.md`](docs/status.md) | Module status and remaining release blockers |

## Development

See [`AGENTS.md`](AGENTS.md) for the project commands, conventions, and
definition of done. The short version:

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

## License

This project is released under the [MIT License](LICENSE).
Copyright 2026 Edgardo Marchi.
