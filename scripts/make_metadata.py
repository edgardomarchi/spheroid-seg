"""CLI wrapper for generating data/metadata.csv from raw-image filenames.

Run after QC and before ``scripts/make_splits.py``. See ``docs/data-pipeline.md``
for the full onboarding flow.
"""

from __future__ import annotations

from spheroid_seg.data.make_metadata import main

if __name__ == "__main__":
    raise SystemExit(main())
