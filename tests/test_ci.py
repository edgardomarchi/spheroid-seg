"""Tests for the GitHub Actions CI workflow file.

These checks validate the workflow definition itself; they do not require a
live GitHub run.
"""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _parse_version(version: str) -> tuple[int, int]:
    """Return (major, minor) for a 'X.Y' or 'X.Y.Z' version string."""
    parts = version.split(".")
    return int(parts[0]), int(parts[1])


def _version_satisfies(version: str, spec: str) -> bool:
    """Check a Python version string against a simple requires-python spec.

    Supports specs of the form ``>=X.Y,<Z.W`` (the current project range).
    """
    v = _parse_version(version)
    lower_match = re.search(r">=\s*(\d+\.\d+)", spec)
    upper_match = re.search(r"<\s*(\d+\.\d+)", spec)
    if lower_match:
        lower = _parse_version(lower_match.group(1))
        if v < lower:
            return False
    if upper_match:
        upper = _parse_version(upper_match.group(1))
        if v >= upper:
            return False
    return True


def _run_commands(workflow: dict) -> list[str]:
    """Collect all shell commands from every job."""
    commands = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run")
            if isinstance(run, str):
                commands.append(run)
    return commands


@pytest.fixture
def workflow() -> dict:
    """Load the CI workflow as a parsed YAML dictionary."""
    assert WORKFLOW_PATH.exists(), f"CI workflow missing: {WORKFLOW_PATH}"
    with WORKFLOW_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def requires_python() -> str:
    """Return the project's ``requires-python`` specifier."""
    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["requires-python"]


def test_workflow_file_exists_and_is_valid_yaml(workflow: dict) -> None:
    """The CI workflow file exists, parses, and has the expected name."""
    assert workflow is not None
    assert workflow.get("name") == "CI"


def test_triggers_cover_push_to_main_and_pull_requests(workflow: dict) -> None:
    """The workflow triggers on pushes to main and on pull requests."""
    # In YAML 1.1 the unquoted key ``on`` is parsed as the boolean ``True``.
    triggers = workflow.get("on") or workflow.get(True)
    assert triggers is not None
    assert "push" in triggers
    assert "pull_request" in triggers
    push = triggers["push"]
    assert isinstance(push, dict)
    assert "main" in push.get("branches", [])


def test_lint_job_runs_ruff_check_and_format_check(workflow: dict) -> None:
    """The lint job runs ruff linting and formatting checks."""
    jobs = workflow.get("jobs", {})
    assert "lint" in jobs
    lint_steps = [step.get("run", "") for step in jobs["lint"].get("steps", [])]
    assert any("ruff check ." in step for step in lint_steps)
    assert any("ruff format --check ." in step for step in lint_steps)


def test_test_job_matrix_is_exactly_supported_python_versions(
    workflow: dict,
    requires_python: str,
) -> None:
    """The test matrix covers exactly 3.12, 3.13, and 3.14."""
    jobs = workflow.get("jobs", {})
    assert "test" in jobs
    matrix = jobs["test"]["strategy"]["matrix"]
    versions = matrix["python-version"]
    assert versions == ["3.12", "3.13", "3.14"]
    for version in versions:
        assert _version_satisfies(version, requires_python), (
            f"Python {version} does not satisfy {requires_python}"
        )


def test_no_job_installs_notebook_group_or_gpu_extras(workflow: dict) -> None:
    """No CI step installs the notebook group or GPU extras."""
    forbidden = [
        "--group notebooks",
        "--all-groups",
        "--extra",
        "[cuda",
        "[rocm",
        "notebooks",
    ]
    for command in _run_commands(workflow):
        for token in forbidden:
            assert token not in command, (
                f"CI command must not reference notebook group or GPU extras: {command!r}"
            )
