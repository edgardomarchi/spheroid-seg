"""Tests for the optional `viz` extra and its error messages."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_viz_extra_exists_and_includes_matplotlib() -> None:
    """pyproject.toml declares a `viz` extra that includes matplotlib."""
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text()
    pyproject = tomllib.loads(pyproject_text)
    extras = pyproject["project"]["optional-dependencies"]

    assert "viz" in extras, "viz extra is missing from project.optional-dependencies"
    assert any("matplotlib" in dep for dep in extras["viz"]), (
        "viz extra does not include matplotlib"
    )


def test_visualize_batches_error_message_mentions_viz_extra() -> None:
    """The lazy-import error in scripts/visualize_batches.py points users to [viz]."""
    script_path = REPO_ROOT / "scripts" / "visualize_batches.py"
    source = script_path.read_text()
    assert "spheroid-seg[viz]" in source, (
        "visualize_batches.py error message should mention pip install spheroid-seg[viz]"
    )
