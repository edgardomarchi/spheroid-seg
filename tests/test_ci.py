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


PIP_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pip-install-check.yml"

# Core stack upper bounds, keyed by package with the excluded "major.minor".
# jax 0.11.0 removed deprecated jax.core internals (e.g. get_opaque_trace_state)
# that the flax line in the lock-proven range still calls; flax and optax are
# capped at the next minor for the same defensive reason. A deliberate stack
# upgrade must update these caps together with the bounds in pyproject.toml.
CORE_STACK_CAPS = {"jax": "0.11", "flax": "0.13", "optax": "0.3"}


def _dependency_spec(dependencies: list[str], package: str) -> str:
    """Return the version specifier for ``package`` within a dependency list.

    Returns an empty string when ``package`` is not declared. Package names are
    matched exactly, so ``jax`` does not match ``jaxlib``.
    """
    for dep in dependencies:
        match = re.match(r"\s*([A-Za-z0-9._-]+)", dep)
        if match and match.group(1).lower() == package.lower():
            return dep[match.end() :].strip()
    return ""


@pytest.fixture
def project_dependencies() -> list[str]:
    """Return the project's runtime dependency list."""
    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["dependencies"]


@pytest.fixture
def pip_workflow() -> dict:
    """Load the pip-install check workflow as a parsed YAML dictionary."""
    assert PIP_WORKFLOW_PATH.exists(), f"pip-install workflow missing: {PIP_WORKFLOW_PATH}"
    with PIP_WORKFLOW_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_dependency_spec_ignores_similar_package_names() -> None:
    """Spec lookup matches package names exactly and tolerates missing entries."""
    deps = ["jaxlib>=0.10.0,<0.11", "scipy"]
    assert _dependency_spec(deps, "jax") == ""
    assert _dependency_spec(deps, "scipy") == ""
    assert _dependency_spec(["jax>=0.10.0,<0.11"], "jax") == ">=0.10.0,<0.11"


@pytest.mark.parametrize("package", sorted(CORE_STACK_CAPS))
def test_core_stack_dependency_is_bounded(package: str, project_dependencies: list[str]) -> None:
    """Core stack deps carry an upper bound below the known-incompatible minor.

    JAX removes deprecated internals on minor bumps, so an unconstrained pip
    install must not float past the uv.lock-proven range (see the comment above
    the core deps in pyproject.toml).
    """
    cap = _parse_version(CORE_STACK_CAPS[package])
    spec = _dependency_spec(project_dependencies, package)
    assert spec, f"{package} is not declared in project.dependencies"
    assert re.search(r">=\s*\d+\.\d+", spec), f"{package} must keep a lower bound: {spec!r}"
    upper = re.search(r"<\s*(\d+\.\d+)", spec)
    assert upper is not None, f"{package} must carry an upper bound: {spec!r}"
    assert _parse_version(upper.group(1)) == cap, (
        f"{package} upper bound must exclude {CORE_STACK_CAPS[package]}: {spec!r}"
    )


def test_pip_workflow_runs_on_packaging_pull_requests_and_weekly(pip_workflow: dict) -> None:
    """The pip job triggers on packaging PRs and on a weekly schedule."""
    triggers = pip_workflow.get("on") or pip_workflow.get(True)
    assert triggers is not None
    paths = triggers.get("pull_request", {}).get("paths", [])
    assert "pyproject.toml" in paths
    assert "uv.lock" in paths
    schedules = triggers.get("schedule", [])
    assert schedules, "a weekly cron schedule is required to catch upstream drift"
    for entry in schedules:
        fields = entry["cron"].split()
        assert len(fields) == 5, f"invalid cron expression: {entry['cron']!r}"
        assert fields[2] != "*" or fields[4] != "*", f"cron must not run daily: {entry['cron']!r}"


def test_pip_workflow_installs_with_pip_not_uv(pip_workflow: dict) -> None:
    """The pip job installs editable with pip, matching the notebook CPU path."""
    jobs = pip_workflow.get("jobs", {})
    assert jobs, "pip-install workflow must define at least one job"
    commands = _run_commands(pip_workflow)
    assert any('pip install -e ".[viz]"' in command for command in commands), (
        'expected the notebook CPU install command: pip install -e ".[viz]"'
    )
    forbidden = ["uv sync", "uv run", "uv pip", "setup-uv", "[cuda", "[rocm", "notebooks"]
    for command in commands:
        for token in forbidden:
            assert token not in command, (
                f"pip-check command must not use uv/lock or GPU extras: {command!r}"
            )


def test_pip_workflow_uses_python_312_and_smoke_checks(pip_workflow: dict) -> None:
    """The pip job pins Python 3.12 and runs import + overfit smoke checks."""
    jobs = pip_workflow.get("jobs", {})
    python_versions = [
        step.get("with", {}).get("python-version")
        for job in jobs.values()
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/setup-python")
    ]
    assert python_versions, "pip-check job must set up Python explicitly"
    assert all(version == "3.12" for version in python_versions)
    commands = _run_commands(pip_workflow)
    assert any("import spheroid_seg" in command for command in commands)
    assert any(
        "spheroid_seg.train" in command and "--overfit-one-batch" in command for command in commands
    )
