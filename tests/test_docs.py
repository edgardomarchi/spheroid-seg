"""Lightweight checks for repo-level documentation and license."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_MIT_LICENSE = """MIT License

Copyright (c) 2026 Edgardo Marchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


@pytest.mark.parametrize("name", ["README.md", "LICENSE"])
def test_required_file_exists(name: str) -> None:
    """README and LICENSE must be present at the repository root."""
    path = REPO_ROOT / name
    assert path.is_file(), f"{name} is missing at repository root"


def test_license_is_canonical_mit() -> None:
    """LICENSE must match the canonical MIT text for 2026 / Edgardo Marchi."""
    license_text = (REPO_ROOT / "LICENSE").read_text()
    assert license_text == CANONICAL_MIT_LICENSE, "LICENSE does not match canonical MIT text"


def test_readme_references_exist() -> None:
    """Every local file linked from README.md must exist."""
    readme = (REPO_ROOT / "README.md").read_text()
    # Markdown links: [text](path)
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    for _text, target in link_pattern.findall(readme):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        # Fragment-only links refer inside the same file.
        if target.startswith("#"):
            continue
        path = REPO_ROOT / target
        assert path.exists(), f"README links to missing path: {target}"
