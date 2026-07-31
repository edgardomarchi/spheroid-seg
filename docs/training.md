# Training pipeline

How a model is trained, from config to checkpoint. For architecture rationale
see `docs/design.md`; for how batches are produced see `docs/data-pipeline.md`.

## Overview

```
configs/*.yaml ──► train.py ──► dataset + patching + augment (data pipeline)
                     │
                     ├─ losses.py    soft Dice + class-weighted cross-entropy
                     ├─ metrics.py   per-class Dice / IoU
                     ├─ model        U-Net (Flax), BatchNorm in 'batch_stats'
                     └─ optimizer    optax AdamW, cosine decay
                     │
                     ▼
        outputs/runs/<config>_<timestamp>/
            ├── logs/train_log.csv      per-epoch losses and Dice
            └── checkpoints/            best-by-mean-val-Dice kept
```

## Components

- **`src/spheroid_seg/losses.py`** — combined soft multiclass Dice +
  class-weighted cross-entropy. Class weights come from the config
  (`class_weights`); background is downweighted so it does not dominate.
- **`src/spheroid_seg/metrics.py`** — per-class Dice and IoU. Documented
  empty-class behavior: true negative scores 1.0, false positive/negative 0.0.
- **`src/spheroid_seg/train.py`** — the training loop:
  - loads a YAML config; no hardcoded hyperparameters;
  - builds the data pipeline, or falls back to the synthetic generator
    (`src/spheroid_seg/data/synthetic.py`) when no real pairs exist in
    `data/raw` / `data/masks`;
  - `flax.training.train_state.TrainState` + `optax.adamw` with cosine decay;
  - BatchNorm statistics are carried in the mutable `batch_stats` collection
    and updated with `train=True`;
  - logs every epoch to CSV and saves checkpoints; the checkpoint with the
    highest mean validation Dice is kept as best.

## Configurations

| Config | Patch | Base features | Purpose |
|---|---|---|---|
| `configs/smoke.yaml` | small | small | fastest sanity checks (overfit-one-batch, CI) |
| `configs/tiny.yaml` | 128 | 16 | short CPU trainings, pipeline verification |
| `configs/base.yaml` | 512 | 32 | full model (7.7M params); intended for GPU |

All tests and smoke checks must pass on CPU-only machines; `base.yaml`
training is expected to run on GPU or Colab.

## Usage

```bash
# full training run
uv run python -m spheroid_seg.train --config configs/base.yaml

# limit epochs (overrides config)
uv run python -m spheroid_seg.train --config configs/tiny.yaml --epochs 20

# optimization sanity check: memorize a single batch
uv run python -m spheroid_seg.train --config configs/smoke.yaml --overfit-one-batch
```

Every invocation writes to a unique `outputs/runs/<config>_<timestamp>/`
directory, so runs never overwrite each other. `outputs/` is gitignored.

## Determinism

Given a fixed seed (in the config), two identical commands must produce
identical losses and metrics, step by step. This is a hard requirement:
experiments are only comparable if randomness is fully seeded. Verify after
any change touching the pipeline:

```bash
uv run python -m spheroid_seg.train --config configs/smoke.yaml --overfit-one-batch
# run twice; initial loss and every step loss must match exactly
```

Note the seed fixes the random *sequence*, not the content: changing code
that consumes randomness (e.g. the synthetic generator) legitimately changes
all downstream numbers.

## Sanity checks and what to expect

- **Overfit-one-batch** (smoke config): loss should fall monotonically to
  near-zero (< 0.05 within ~100 steps). If it does not, the optimization
  path (gradients, BatchNorm wiring, loss) is broken.
- **Short synthetic training** (tiny config, ~20 epochs): the synthetic task
  is trivially learnable; per-class validation Dice should exceed 0.9 within
  a few epochs. Best checkpoint may come from an early epoch — later epochs
  are noisier due to the small validation set and strong augmentation.

## Known limitations

- The full `base.yaml` overfit-one-batch check is impractical on CPU; run it
  on GPU/Colab when available.
- Real-data training requires `data/splits/*.txt` and the leak check between
  train/val; the synthetic fallback exists only for smoke testing.
