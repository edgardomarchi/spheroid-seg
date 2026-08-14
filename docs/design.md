# Design Doc — Spheroid & Cell Segmentation in Phase-Contrast Microscopy

> Living design document. Last updated: 2026-07-31.
> Intended use: reference spec for the open-source repo. Kimi Code (or any coding agent)
> consumes it via a short root-level `AGENTS.md` that points here (see §9).

---

## 1. Goal

Build a pipeline that, from phase-contrast microscopy images (4x and 10x):

1. **Semantically segments**: background / loose cell / cell aggregate.
2. **Separates individual instances** (each cell, each aggregate). *(v0.2)*
3. **Classifies each aggregate** as spheroid or organoid. *(v0.2)*
4. **Extracts per-instance morphometrics**: area, equivalent diameter, circularity,
   solidity, eccentricity. *(v0.2)*

Constraints: modest compute (single modest GPU), small proprietary dataset
(~80–100 annotated images initially, scaling toward ~400), framework **JAX/Flax**
(deliberate choice — developer learning goal). The repository will be **open source**.

### Release scope

| Release | Contents |
|---|---|
| **v0.1** | Semantic segmentation only: data pipeline, U-Net (Flax, from scratch), training, evaluation, full-image inference via patch stitching |
| **v0.2** | Instance separation (watershed), morphometrics, spheroid/organoid classification (hybrid, see §4 D2) |
| Deferred (M7) | SLiMIA pre-training fallback if the v0.1 baseline underperforms |

---

## 2. Data

### 2.1 Sources

| Source | Status | Use |
|---|---|---|
| In-house images annotated by the translational medicine group | ~80–100 within ~1 week; scaling to ~400 | Main train/val/test |
| SLiMIA (figshare; Nature Sci Data s41597-025-04441-x) — ~8,000 bright-field/phase-contrast spheroid images, 9 microscopes, 47 cell lines | Public | Pipeline exploration and possible pre-training. **Caveat**: spheroids larger and denser than ours → domain shift; never use as validation ground truth for our domain |
| 6 sample images (3 raw/annotated pairs) | Available now | Data-pipeline smoke tests only |

### 2.2 Annotation format (spec for the clinical group)

- **One PNG per image**, same base name and dimensions as the source image.
- Pixel value = class ID: `0` = background, `1` = loose cell, `2` = spheroid, `3` = organoid.
- Lossless grayscale PNG (uint8). Never JPEG for masks.
- **File naming**: base names must end in `_4x` or `_10x` (e.g. `117 - 9d 3T3_4x.JPG`)
  so magnification can be parsed for stratified splitting and per-magnification
  metrics. Alternatively, magnification can be supplied via `data/metadata.csv`;
  the filename suffix takes precedence.
- **Export script**: `scripts/export_qupath_masks.groovy` rasterizes QuPath
  annotations into spec-compliant masks (see `docs/data-pipeline.md` for usage
  and annotation pitfalls).
- Suggested tools: QuPath (preferred), napari, Fiji/ImageJ.
- **Annotation protocol**: written spheroid-vs-organoid definitions with an example
  gallery including agreed borderline cases. *Pending: validate objective criteria
  with the clinical group (see §10).*

### 2.3 Directory layout

```
data/
  raw/                # original images (JPG/TIFF) — never modified
  masks/              # annotation PNGs (IDs 0-3)
  slimia/             # external dataset, if used
  splits/             # train.txt / val.txt / test.txt (image-level split)
outputs/
  checkpoints/        # model weights
  logs/               # per-epoch metrics (CSV)
  predictions/        # predicted masks on val/test
  metrics/            # morphometry tables (v0.2)
```

### 2.4 Data distribution (open-source policy)

- **The code repo contains no images.** All of `data/` is `.gitignore`d; only its
  directory structure and `data/splits/*.txt` are committed.
- **Public sample subset**: subject to clinical group approval, a small anonymized
  set (e.g. the 6 sample images) is published on Zenodo/figshare with a DOI, plus a
  `scripts/download_data.py` fetch script — the standard practice in bioimage papers.
- **Full dataset**: private, available on request (documented in README). If real
  data versioning alongside code is ever needed, use DVC (or a git-lfs-backed data
  repo + submodule) — never plain git for GBs of images.

### 2.5 Split rules

- **Image-level split, never patch-level**: 70% train / 15% val / 15% test.
- Stratify by magnification (4x/10x) and, if applicable, by cell line/condition.
- Files under `data/splits/` are the single source of truth (reproducibility).

---

## 3. Pipeline architecture

```
Image → [Stage 1: semantic U-Net]   → 3-class mask          (v0.1)
      → [Stage 2: instance split]   → per-object label map  (v0.2)
      → [Stage 3: morphometrics + classification] → metrics table (v0.2)
```

### Stage 1 — Semantic segmentation (U-Net in Flax) — v0.1

- **Model classes: 3** (background / loose cell / aggregate). The spheroid/organoid
  distinction is NOT made by the network (see §4 D2).
- Standard U-Net from scratch (encoder-decoder with skip connections, ~120–150 lines):
  - Conv block: 2× (Conv 3×3 → BatchNorm → ReLU).
  - Downsampling: 2×2 max-pool. 4 levels (base features: 32 or 64, VRAM-dependent).
  - Upsampling: 2×2 ConvTranspose + skip concatenation + conv block.
  - Output: 1×1 Conv → 3 channels (logits).
- Input: **512×512 patches** (fallback 256×256 if VRAM is tight), extracted with
  oversampling of object-containing regions (avoid 95% background-only patches).
- Preprocessing: per-image intensity normalization to percentiles 1–99, scaled to
  [0, 1]. Grayscale vs RGB decided empirically at baseline (config flag).
- **No physical-size normalization** (the images' scale bars are incorrect). Both
  magnifications handled by: single model + scale augmentation (zoom 0.5×–2×).
- Full-image inference: overlapping patches (~10–20% overlap) with logit averaging
  in overlap regions.

### Stage 2 — Instance separation — v0.2

- Per object class (cell, aggregate):
  - `scipy.ndimage.label` for well-separated objects.
  - Where objects touch: distance transform + **watershed** (skimage) with seeds
    from distance peaks.
- Mask smoothing before measuring (light morphological opening/closing or Gaussian +
  re-binarization): circularity is highly sensitive to jagged borders.
- Filter instances by minimum area (debris/noise; threshold calibrated on val).

### Stage 3 — Morphometrics + spheroid/organoid classification — v0.2

- Per-instance metrics via `skimage.measure.regionprops`: area, equivalent diameter,
  perimeter, **circularity = 4π·area / perimeter²**, solidity, eccentricity, bbox.
- Aggregate classification — **hybrid strategy** (two implementations, compared on
  validation against clinical labels):
  1. **Morphometric rule** (e.g. high circularity+solidity → spheroid; low → organoid).
     Thresholds tunable with the clinical group without retraining.
  2. **Per-instance classifier**: small CNN on instance crops, or morphometric
     features + classical classifier (e.g. gradient boosting).
- Output: per-image CSV (one row per instance) + summary (counts, distributions).

---

## 4. Design decisions

| # | Decision | Rejected alternative | Rationale |
|---|---|---|---|
| D1 | Semantic segmentation + post-processing | End-to-end instance segmentation (Mask R-CNN etc.) | Less compute, more robust with small data |
| D2 | 3-class network; spheroid/organoid by **hybrid** rule + instance classifier (v0.2) | 4-class end-to-end | The spheroid/organoid boundary is subjective; keep it out of pixel-level training, compare rule vs classifier empirically |
| D3 | U-Net from scratch in pure Flax | Pretrained ImageNet encoder (e.g. via Keras 3 JAX backend); Cellpose fine-tune; nnU-Net | Learning goal (JAX); Cellpose underperformed in a quick test; 80–100 images + strong augmentation suffice for a small U-Net |
| D4 | Single model for 4x and 10x + scale augmentation | Two per-magnification models; magnification input channel | Simplicity; escalate to two models only if the single one underperforms |
| D5 | SLiMIA as deferred fallback (M7) | Pre-train from day 1; ignore entirely | Domain shift (larger, denser spheroids; different dominant modality) — keep it off the critical path |
| D6 | v0.1 = segmentation only | Full pipeline in v0.1 | Smaller, testable open-source release; instances/morphometrics build on validated masks |

**Escape hatch for D2**: the model must switch to 4-class output by changing only
`num_classes` in config, in case spheroid/organoid annotations prove highly
consistent and the hybrid approach disappoints.

**Status notes (2026-08-14)**

- **D2**: the first real-data baseline shows that loose↔aggregate cross-confusion
  dominates the error while a GT audit found zero mixed-class objects and no
  annotation noise. This supports the escape-hatch intuition that the
  loose/aggregate distinction may be better handled by v0.2 object-level
  classification than by the pixel-level network. Decision deferred until
  ~50 annotated images: either stay 3-class or collapse to 2-class
  (background / object). See `docs/status.md`.
- **D4**: no per-magnification failure signal so far (per-image background Dice
  10x ≈ 4x in the 3-image validation split), but the evidence is weak because val n
  is 2 vs 1. The single-model strategy stays; revisit after the full batch.

---

## 5. Tech stack

- **Python**: `requires-python = ">=3.12,<3.15"` in `pyproject.toml` (a range, not a
  pin — this is a library-style open repo). **3.12 is the reference version** used
  for development and CI; CI also tests 3.13 and 3.14 in a matrix. Rationale: core
  stack (JAX, scikit-image, etc.) only gained full 3.14 wheel support in late 2025,
  so 3.12 is the safe, fully-verified baseline.
- **uv** for dependency and environment management, `pyproject.toml`
  as the single project manifest, `src/` layout.
- Core: `jax`, `flax`, `optax`.
- Data/vision: `albumentations`, `opencv-python-headless`, `scikit-image`, `scipy`,
  `tifffile`, `numpy`, `pyyaml`.
- Dev group: `pytest`, `ruff` (lint + format), `mypy` (optional).
- Notebook group: `jupyter`, `matplotlib` (QC/exploration).
- Logging: per-epoch CSV (keep v0.1 dependency-light; TensorBoard optional).

### Hardware strategy

**CPU-first development**: all tests, smoke checks, and short synthetic trainings
must pass on CPU-only machines using `configs/smoke.yaml` / `configs/tiny.yaml`.
GPU is never assumed available and never required for development or CI.

Full-model training (`configs/base.yaml`, 512² patches, 7.7M params) targets a
free cloud GPU tier (e.g. Colab T4 16 GB) or any on-demand local CUDA GPU with
~8 GB VRAM. A local ROCm iGPU may serve as a light-GPU option, but JAX-on-ROCm
support must be verified (`jax.devices()`) before relying on it.

Consequences:

- Cloud workflow: clone repo + `uv sync`, upload `data/` out-of-band (e.g.
  Drive), run training, download checkpoints. Cloud sessions can be cut at any
  time, so checkpoints must be saved frequently and training must be resumable.
  This also doubles as a clean-room reproducibility test of the repo.
- The full `base.yaml` overfit-one-batch acceptance check is impractical on
  CPU; run it once on GPU.

```bash
uv sync --all-groups                 # full dev setup
uv run pytest                        # tests
uv run python -m spheroid_seg.train --config configs/base.yaml
uv run python -m spheroid_seg.eval  --config configs/base.yaml
```

---

## 6. Repository layout

```
spheroid-seg/
  AGENTS.md                  # short agent context (see §9)
  pyproject.toml             # uv-managed manifest
  README.md                  # quickstart, citing this design doc
  docs/design.md             # THIS document
  docs/data-pipeline.md      # data pipeline walkthrough
  docs/training.md           # training pipeline walkthrough
  docs/status.md             # module progress and pending items
  configs/base.yaml          # num_classes, patch_size, lr, features, paths, seed
  configs/tiny.yaml          # small config for CPU smoke trainings
  configs/smoke.yaml         # minimal config for CI / overfit-one-batch checks
  src/spheroid_seg/
    data/                    # dataset, patching, augmentation, splits
    models/unet.py           # Flax U-Net
    train.py                 # training loop (optax)
    eval.py                  # Dice/IoU, confusion, overlays
    infer.py                 # full-image inference (patch stitching)
    postproc/                # (v0.2) instances, morphometrics, classification
  notebooks/                 # exploration (SLiMIA, annotation QC)
  tests/
  data/  outputs/            # per §2.3 (contents not committed)
```

---

## 7. Training

- **Loss**: Dice + class-weighted Cross-Entropy (low weight on background — heavy
  class imbalance).
- **Optimizer**: AdamW, lr 1e-3 with cosine decay, weight decay 1e-4.
- **Batch size**: largest that fits VRAM (estimate 8–16 patches @ 512²).
- **Augmentation**: flips, 90° rotations, elastic, zoom 0.5–2×, Gaussian blur,
  noise, and strong brightness/contrast jitter (illumination varies a lot across
  fields).
- **Early stopping** on validation Dice (patience ~20 epochs).
- **Fixed seed**, per-epoch metrics logged to CSV.
- **Metrics**: per-class Dice and IoU (never global accuracy — 95% background
  inflates it). Instance-level confusion for cell/aggregate (v0.2), plus
  spheroid/organoid agreement (v0.2).

---

## 8. Evaluation and success criteria

1. **Quantitative**: per-class Dice on test. Initial orientation targets: aggregate
   Dice ≥ 0.85, cell Dice ≥ 0.75 (revisit after baseline).
2. **Instance level** (v0.2): detection F1 (IoU ≥ 0.5 matching between predicted
   and annotated instances).
3. **Morphometrics** (v0.2): correlation between metrics computed on annotated vs
   predicted masks (same images). This is the validation the clinical group cares
   about.
4. **Qualitative**: overlay grids on test before accepting any checkpoint.

### Current baseline (2026-08-14)
Baseline computed on the validation split — the same split that guided early stopping, so these numbers are mildly optimistic. The test split remains untouched and is reserved for the final v0.1 acceptance check (§8 criteria).
First real-data training: `outputs/runs/colab_20260812_182023`,
`configs/base.yaml` with `bn_momentum: 0.9`, early stopping at epoch 47,
best checkpoint at ~epoch 27. Evaluated on the 3-image validation split
(2× 10x, 1× 4x). **All numbers are post-fix** for the float32 confusion-matrix
saturation bug (`outputs/evals/base_20260814_132521/`).

Validation Dice (background / loose cell / aggregate):

| Group | n | background | loose cell | aggregate |
|---|---|---:|---:|---:|
| overall | 3 | 0.9907 | 0.3554 | 0.2954 |
| 10x | 2 | 0.9888 | 0.3505 | 0.2983 |
| 4x | 1 | 0.9945 | 0.3664 | 0.2856 |

Confusion matrix (rows = GT, columns = prediction; total = 150,160,512 px):

| GT \ pred | background | loose cell | aggregate |
|---|---:|---:|---:|
| background | 145,127,942 | 351,148 | 2,236,410 |
| loose cell | 70,879 | 434,604 | 426,133 |
| aggregate | 61,393 | 728,261 | 723,742 |

Interpretation: background is essentially solved; the model struggles to
discriminate loose cells from aggregates at the pixel level, with roughly
half of each foreground class mislabelled as the other. A GT audit found no
mixed-class objects and no annotation noise, so this is a genuine semantic
limitation. See `docs/status.md` for the full baseline, decision evidence,
and next steps.

---

## 9. Working with coding agents (Kimi Code)

- **Root `AGENTS.md`** — short (30–60 lines), stable, with pointers and validation
  commands. Minimum content:

```markdown
# AGENTS.md
Project: phase-contrast spheroid/cell segmentation (JAX/Flax). Open source.
Full design spec: read docs/design.md before any task.
## Commands
- Setup: `uv sync --all-groups`
- Tests: `uv run pytest`
- Train: `uv run python -m spheroid_seg.train --config configs/base.yaml`
- Eval:  `uv run python -m spheroid_seg.eval --config configs/base.yaml`
## Conventions
- Single source of config in configs/*.yaml; no hardcoded hyperparameters.
- Data splits only from data/splits/*.txt (never re-shuffle).
- Masks use IDs 0-3 (background/cell/spheroid/organoid); the model trains with
  3 classes (2 and 3 merged as "aggregate").
- English for all code, comments, and docs.
```

- **Modular prompts** (one module per prompt, with acceptance criteria):
  - M1: data pipeline (raw+mask loading, annotation QC, patching, augmentation).
    *Acceptance: visualize 8 augmented batches + class stats.*
  - M2: Flax U-Net + shape tests. *Acceptance: forward pass on a 512² patch yields
    512²×3 logits.*
  - M3: training loop + logging + checkpoints. *Acceptance: overfits 1 batch
    (loss → ~0).*
  - M4: evaluation (Dice/IoU, overlays). *Acceptance: val report with visual grid.*
  - M5: full-image stitching inference. *Acceptance: full mask for one 4x and one
    10x image.*
  - M6 (v0.2): post-processing (instances + watershed + morphometrics + hybrid
    classification). *Acceptance: metrics CSV on ANNOTATED val masks.*
  - M7 (deferred): SLiMIA pre-training if the baseline underperforms.
- Architecture decisions are made outside the coding agent; prompts implement
  decisions already taken.

---

## 10. Open items and risks

**Pending with the clinical group**
- [ ] Objective spheroid-vs-organoid criteria (circularity? lumen? size?) + gallery
  of borderline examples.
- [ ] Confirm annotation delivery format (§2.2).
- [ ] Which final metrics they need and in which units (scale bars are wrong →
  agree on a real µm/px calibration per magnification, or report in pixels).

**Risks**

| Risk | Mitigation |
|---|---|
| Inconsistent class 2/3 annotations | D2 keeps that distinction out of segmentation training |
| Few images (80–100) | Aggressive augmentation, small U-Net, early stopping; escalate to SLiMIA pre-training (M7) |
| Low contrast in 4x / background noise in 10x | Percentile normalization, brightness/contrast + noise augmentation |
| Insufficient VRAM | 256² patches, base features 32, gradient accumulation |
| Incorrect scale bars | Report in pixels until a reliable per-magnification µm/px calibration exists |
