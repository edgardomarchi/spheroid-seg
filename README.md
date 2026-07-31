# spheroid-seg

Open-source pipeline for segmenting spheroids and loose cells in phase-contrast microscopy images (4x and 10x). Version 0.1 focuses on semantic segmentation (background / loose cell / aggregate) using a U-Net built in JAX/Flax. See [`docs/design.md`](docs/design.md) for the full design specification, data format, architecture decisions, and roadmap.

## Setup

This project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
```

## Usage

Commands below are stubs for the upcoming training and evaluation loops:

```bash
uv run python -m spheroid_seg.train --config configs/base.yaml
uv run python -m spheroid_seg.eval  --config configs/base.yaml
uv run python -m spheroid_seg.infer --config configs/base.yaml
```

## Data

The repository does **not** contain images. A small public sample set will be published on Zenodo and fetched via `scripts/download_data.py`; the full annotated dataset is private and available on request. Splits are committed under `data/splits/*.txt` as the single source of truth.
