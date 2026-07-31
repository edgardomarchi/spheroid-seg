# AGENTS.md

Open-source pipeline for segmenting spheroids and loose cells in phase-contrast
microscopy images (4x and 10x). Framework: JAX/Flax. v0.1 scope: semantic
segmentation only (background / loose cell / aggregate).

**Read `docs/design.md` before any non-trivial task** — it is the full design
spec (data formats, architecture, decisions, risks). This file only holds stable
conventions and commands; do not duplicate design content here.

## Stack

- Python `>=3.12,<3.15` (dev may run any version in the range; CI must cover the
  minimum, 3.12, plus the latest), uv-managed (`pyproject.toml`, `src/` layout).
- Core: `jax`, `flax`, `optax`. Data/vision: `albumentations`, `scikit-image`,
  `scipy`, `opencv-python-headless`, `tifffile`.
- Tests/lint: `pytest`, `ruff`.

## Commands

- Setup: `uv sync --all-groups`
- Tests: `uv run pytest`
- Lint/format: `uv run ruff check . && uv run ruff format .`
- Train: `uv run python -m spheroid_seg.train --config configs/base.yaml`
- Eval: `uv run python -m spheroid_seg.eval --config configs/base.yaml`

## Conventions

- All code, comments, commits, and docs in **English**.
- Single source of configuration in `configs/*.yaml`; never hardcode
  hyperparameters (patch size, lr, features, paths, seed).
- Data splits come only from `data/splits/*.txt` (image-level, stratified by
  magnification). Never re-shuffle or split at patch level.
- Masks are grayscale PNGs with class IDs 0-3 (background / loose cell /
  spheroid / organoid), same base name and size as the raw image.
  The model trains with **3 classes**: IDs 2 and 3 are merged as "aggregate".
- Images' scale bars are known to be incorrect: report sizes in pixels; never
  convert to physical units.
- Files under `data/raw/` are read-only originals. Generated artifacts go to
  `outputs/` and are never committed.
- **No images in the repo**: all of `data/` is `.gitignore`d (except
  `data/splits/*.txt`). A small public sample set lives on Zenodo and is fetched
  via `scripts/download_data.py`; the full dataset is private, on request. Never
  commit image files, and never add them to git-lfs or submodules without an
  explicit decision recorded in `docs/design.md`.

## Testing rules

- Tests encode the spec (docs/design.md + the task's acceptance criteria).
  NEVER weaken, delete, skip, or relax an assertion to make a failing test
  pass. If a test contradicts the implementation, fix the implementation —
  or stop and escalate the conflict in the session report.
- Any modification to an EXISTING test must be flagged and justified in the
  session report (what changed, why the old assertion was wrong).
- Each acceptance criterion from the task prompt maps to at least one test;
  include negative tests (invalid inputs must be rejected, not just valid
  ones accepted).

## Git workflow

- **The agent NEVER runs git write operations** (`add`, `commit`, `push`,
  `merge`, `rebase`, `tag`, creating or switching branches). All git writes are
  executed by the human maintainer. Read-only inspection (`git status`,
  `git diff`, `git log`) is allowed.
- When finishing a task, the agent may SUGGEST a commit message but must not
  run it.
- Branching (human-run): one `feat/<module>` / `fix/<topic>` / `docs/<topic>`
  branch per module, merged to `main` when green; `main` must always stay green.
- Commit style: Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`,
  `test:`).

## Session reports

At the end of every task, write a Markdown report to `outputs/reports/` with a
file name representative of the actions taken (e.g. `m1-data-pipeline.md`,
`fix-config-loading.md`). The report must list: files created/modified, commands
run with their outputs (acceptance criteria), decisions made, and any deviations
from this file or `docs/design.md`.

## Definition of done for any change

1. `uv run pytest` green.
2. `uv run ruff check .` clean.
3. New behavior covered by a test or an explicit acceptance check stated in the
   task prompt.
4. Config defaults in `configs/base.yaml` updated if the change adds parameters.