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
    and updated with `train=True`; the momentum is exposed as `bn_momentum`
    in the config so running statistics can be tuned for small batches.
  - logs every epoch to CSV and saves checkpoints; the checkpoint with the
    highest mean validation Dice is kept as best.

## Configurations

| Config | Patch | Base features | BN momentum | Purpose |
|---|---|---|---|---|
| `configs/smoke.yaml` | small | small | 0.9 | fastest sanity checks (overfit-one-batch, CI) |
| `configs/tiny.yaml` | 128 | 16 | 0.9 | short CPU trainings, pipeline verification |
| `configs/base.yaml` | 512 | 32 | 0.9 | full model (7.7M params); intended for GPU |

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

## Cloud GPU (Colab)

`configs/base.yaml` (512² patches, 7.7M parameters) is impractical on CPU. A
free Colab T4 runtime is enough to run it, but the default `batch_size: 8` can
push against the T4's ~15 GiB VRAM during XLA compilation. The notebook
therefore uses `configs/colab.yaml` for full training.

Open the notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edgardomarchi/spheroid-seg/blob/main/notebooks/colab_training.ipynb)

What the notebook does, in order:

1. Detects whether a GPU is available (`nvidia-smi`) and clones the repo.
2. Installs the package with pip in editable mode, adding the `viz` extra and,
   on GPU runtimes, the `cuda12` JAX extra:
   `pip install -e ".[cuda12,viz]"` (GPU) or `pip install -e ".[viz]"` (CPU).
   The install cell skips if the package is already importable.
3. Runs a fast post-install sanity check: import `spheroid_seg`, print the
   package version, and `jax.devices()`.
4. Optionally loads a private `data/` directory from Google Drive (see below);
   by default this step is skipped and the synthetic fallback is used.
5. Runs the pending M3 acceptance check:
   `python -m spheroid_seg.train --config configs/base.yaml --overfit-one-batch`
   on GPU only — loss should fall to near-zero. The subprocess sets
   `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to leave headroom for XLA.
6. Runs a few epochs of full training with `configs/colab.yaml` (same model as
   `base.yaml` but `batch_size: 4`) on the synthetic fallback to measure GPU
   throughput. This subprocess also sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`.
7. Plots training curves from the run's CSV log.
8. Zips the latest run under `outputs/runs/colab_*/` and downloads it, because
   Colab sessions can be cut at any time.

### Optional Google Drive data loading

For real-data training, set `USE_DRIVE_DATA = True` in the setup cell. The
notebook will then mount Drive and copy `raw/`, `masks/`, and `splits/` from
`DRIVE_DATA_DIR` (default: `/content/drive/MyDrive/Colab Notebooks/spheroid-seg/data`)
into the repo's `data/` directory using `shutil.copytree(..., dirs_exist_ok=True)`.
Paths containing spaces are handled without shelling out.

Leave `USE_DRIVE_DATA = False` (default) to keep the built-in synthetic fallback
and ensure no private data is uploaded or committed. `data/` is `.gitignore`d
(except `data/splits/*.txt`), so copied images are never committed to git.

After copying, the cell prints the number of files in `data/raw` and
`data/masks` and warns if either is zero, so a missing Drive path no longer
silently falls back to synthetic data.

### GPU memory note

All training subprocesses launched by the notebook set
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`. This caps JAX/XLA at 90 % of GPU memory
and prevents the allocator from repeatedly trying to grab the whole 16 GiB
budget on a T4. `configs/colab.yaml` further halves activation memory with
`batch_size: 4` while keeping the same 512² patches and base_features=32 as
`configs/base.yaml`.

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
