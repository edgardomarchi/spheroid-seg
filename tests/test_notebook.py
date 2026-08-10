"""Validation for the Colab training notebook."""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "colab_training.ipynb"


def _notebook() -> dict:
    """Load the notebook as a JSON object."""
    assert NOTEBOOK_PATH.is_file(), f"Notebook not found: {NOTEBOOK_PATH}"
    return json.loads(NOTEBOOK_PATH.read_text())


def _code_sources() -> list[tuple[int, str]]:
    """Return (cell_index, joined_source) for every code cell."""
    return [
        (idx, "".join(cell["source"]))
        for idx, cell in enumerate(_notebook()["cells"])
        if cell["cell_type"] == "code"
    ]


def test_notebook_exists_and_is_valid_json() -> None:
    """The Colab notebook exists and is valid JSON."""
    nb = _notebook()
    assert "cells" in nb
    assert "metadata" in nb
    assert nb.get("nbformat") == 4


def test_notebook_uses_python3_kernelspec() -> None:
    """The notebook kernelspec is the stock Python 3 kernel."""
    nb = _notebook()
    kernelspec = nb["metadata"].get("kernelspec", {})
    assert kernelspec.get("name") == "python3"
    assert kernelspec.get("language") == "python"


def test_notebook_cells_have_cleared_outputs() -> None:
    """All code cells have empty outputs and no execution count."""
    nb = _notebook()
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs") == [], f"Cell {idx} has non-empty outputs"
        assert cell.get("execution_count") is None, f"Cell {idx} has execution_count"


def test_notebook_code_cells_compile() -> None:
    """Every code cell source compiles as Python."""
    nb = _notebook()
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"<notebook-cell-{idx}>", "exec")


def test_notebook_references_repository_url() -> None:
    """The notebook references the repository URL declared in pyproject.toml."""
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text()
    pyproject = tomllib.loads(pyproject_text)
    repo_url = pyproject["project"]["urls"]["Repository"]
    nb_text = NOTEBOOK_PATH.read_text()
    assert repo_url in nb_text, f"Notebook does not reference {repo_url}"


def test_notebook_has_gpu_check_and_uv_commands() -> None:
    """The notebook contains the expected Colab workflow markers."""
    nb = _notebook()
    sources = "".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    assert "nvidia-smi" in sources
    assert "uv" in sources
    assert "sync" in sources
    assert "jax.devices()" in sources
    assert "overfit-one-batch" in sources


def test_notebook_has_gpu_detection_logic() -> None:
    """The notebook detects GPU presence and selects the JAX extra accordingly."""
    sources = "".join(source for _idx, source in _code_sources())
    assert "HAS_GPU" in sources
    assert "cuda12" in sources


def test_notebook_uses_path_based_uv() -> None:
    """No code cell hardcodes a home-relative uv binary path."""
    nb_text = NOTEBOOK_PATH.read_text()
    assert "~/.cargo/bin/uv" not in nb_text
    assert "~/.local/bin/uv" not in nb_text


def _is_absolute_path_expression(node: ast.AST) -> bool:
    """Return True if the AST node represents an absolute filesystem path."""
    # String literal starting with "/".
    is_abs_literal = (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("/")
    )
    # str(REPO) or str(Path(...)).
    is_str_call = (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and bool(node.args)
    )
    return is_abs_literal or is_str_call


def test_notebook_subprocess_calls_use_explicit_cwd_or_absolute_paths() -> None:
    """Every repo-level subprocess.run passes cwd=REPO or uses absolute paths."""
    for idx, source in _code_sources():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr == "run"
            ):
                continue

            has_cwd = any(isinstance(kw, ast.keyword) and kw.arg == "cwd" for kw in node.keywords)
            if has_cwd:
                continue

            # Calls that do not operate inside the repo are allowed to omit cwd.
            first_arg = node.args[0] if node.args else None
            if isinstance(first_arg, ast.List) and first_arg.elts:
                cmd_parts = [
                    elt.value
                    for elt in first_arg.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
                if "nvidia-smi" in cmd_parts:
                    continue
                if "-m" in cmd_parts and "pip" in cmd_parts:
                    continue

            # Calls that use absolute paths for all repo-level arguments are allowed.
            if any(_is_absolute_path_expression(arg) for arg in node.args):
                continue
            first_arg = node.args[0] if node.args else None
            if isinstance(first_arg, ast.List) and any(
                _is_absolute_path_expression(elt) for elt in first_arg.elts
            ):
                continue

            raise AssertionError(
                f"Cell {idx}: subprocess.run lacks cwd=REPO and is not an exempt call"
            )


def _subprocess_run_calls(source: str) -> list[ast.Call]:
    """Return all subprocess.run(...) calls in the source."""
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
            and func.attr == "run"
        ):
            calls.append(node)
    return calls


def _call_targets_nvidia_smi(node: ast.Call) -> bool:
    """Return True if the subprocess.run call is the GPU detection check."""
    first_arg = node.args[0] if node.args else None
    if not isinstance(first_arg, ast.List):
        return False
    cmd_parts = [
        elt.value
        for elt in first_arg.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    ]
    return "nvidia-smi" in cmd_parts


def test_notebook_command_cells_use_streaming_helper() -> None:
    """All command invocations go through the streaming run() helper.

    The only allowed subprocess.run call is the one-off GPU detection in the
    setup cell; everything else must use the helper so output is streamed live.
    """
    for idx, source in _code_sources():
        for node in _subprocess_run_calls(source):
            assert _call_targets_nvidia_smi(node), (
                f"Cell {idx}: bare subprocess.run call found; "
                "route repo commands through the run() helper"
            )


def test_notebook_defines_streaming_run_helper() -> None:
    """The notebook defines a streaming run() helper that uses subprocess.Popen."""
    sources = "".join(source for _idx, source in _code_sources())
    assert "def run(" in sources
    assert "subprocess.Popen" in sources


def test_notebook_plot_cell_reads_train_log_csv() -> None:
    """The training-curves cell reads the per-run train_log.csv."""
    sources = "".join(source for _idx, source in _code_sources())
    assert "train_log.csv" in sources
    assert "matplotlib.pyplot" in sources
