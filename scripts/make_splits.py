"""CLI wrapper for creating stratified train/val/test split files from real data.

Run after QC and after filling ``data/metadata.csv``. See ``docs/data-pipeline.md``
for the full onboarding flow.
"""

from __future__ import annotations

from spheroid_seg.data.make_splits import main

if __name__ == "__main__":
    raise SystemExit(main())
