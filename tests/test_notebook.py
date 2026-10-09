"""Validation for the Colab training notebook.

The notebook is an explicit experiment runner: a settings-only first code cell,
a derivation/helper cell, a resolved execution plan gate, repository
synchronization, a JAX/CUDA install-repair cell, a post-install JAX sanity gate,
and then the optional Drive data step, the optional overfit-one-batch check,
the main training cell, and artifact inspection/persistence. These tests pin
that structure without requiring Colab or a GPU.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "colab_training.ipynb"

# User-facing parameters that must be assigned in the first code cell.
REQUIRED_SETTINGS = [
    "GIT_REPOSITORY_URL",
    "GIT_REF",
    "REPO_PATH",
    "USE_DRIVE_DATA",
    "DRIVE_DATA_DIR",
    "DRIVE_RUNS_DIR",
    "TRAIN_CONFIG",
    "TRAIN_EPOCHS",
    "RUN_OVERFIT_CHECK",
    "FORCE_FRESH",
    "ALLOW_DIRTY_REPO",
    "REPAIR_JAX_ENVIRONMENT",
]

# Values derived from the settings or probed from the runtime; they belong in
# the second code cell, never in the settings cell.
DERIVED_NAMES = [
    "HAS_GPU",
    "PIP_EXTRA",
    "CONFIG_STEM",
    "TRAIN_PREFIX",
    "DRIVE_CONFIG_PATH",
    "def run(",
]

# git subcommands allowed as bare subprocess.run calls (read-only inspection).
READ_ONLY_GIT_SUBCOMMANDS = {"branch", "describe", "diff", "log", "rev-parse", "status"}


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


def _settings_source() -> str:
    """Return the source of the first code cell: user settings only."""
    return _code_sources()[0][1]


def _second_code_cell_source() -> str:
    """Return the source of the second code cell: derivation and helpers."""
    return _code_sources()[1][1]


def _markdown_sources() -> list[str]:
    """Return the joined source of every markdown cell."""
    return [
        "".join(cell["source"]) for cell in _notebook()["cells"] if cell["cell_type"] == "markdown"
    ]


def _cell_index_containing(*markers: str) -> int:
    """Return the index of the first code cell containing all given markers."""
    for idx, source in _code_sources():
        if all(marker in source for marker in markers):
            return idx
    raise AssertionError(f"No code cell contains all markers: {markers}")


def _plan_cell_source() -> str:
    """Return the source of the resolved execution plan cell."""
    return dict(_code_sources())[_cell_index_containing("Resolved execution plan")]


def _repo_sync_cell_source() -> str:
    """Return the source of the repository synchronization cell."""
    return dict(_code_sources())[_cell_index_containing("git", "clone", "fetch")]


def _install_cell_source() -> str:
    """Return the source of the environment install/repair cell."""
    return dict(_code_sources())[_cell_index_containing("pip", "importlib.metadata")]


def _jax_gate_cell_source() -> str:
    """Return the source of the post-install JAX sanity gate cell."""
    return dict(_code_sources())[_cell_index_containing("import jax", "jax.devices()")]


def _overfit_cell_source() -> str:
    """Return the source of the optional overfit-one-batch cell."""
    return dict(_code_sources())[_cell_index_containing("overfit-one-batch", "Skipping")]


def _drive_code_cell_index() -> int:
    """Return the index of the Drive data-loading code cell."""
    return _cell_index_containing("USE_DRIVE_DATA", "shutil.copytree")


def _first_training_cell_index() -> int:
    """Return the index of the first code cell that invokes training."""
    return _cell_index_containing("spheroid_seg.train")


def _drive_resume_cell_index() -> int:
    """Return the index of the training cell implementing Drive resume."""
    return _cell_index_containing("--resume", "spheroid_seg.train")


def _drive_resume_cell_source() -> str:
    """Return the source of the training cell implementing Drive resume."""
    return dict(_code_sources())[_drive_resume_cell_index()]


def test_notebook_exists_and_is_valid_json() -> None:
    """The Colab notebook exists and is valid JSON with nbformat 4 metadata."""
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


def test_notebook_has_gpu_check_and_pip_commands() -> None:
    """The notebook contains the expected Colab workflow markers."""
    sources = "".join(source for _idx, source in _code_sources())
    assert "nvidia-smi" in sources
    assert "pip" in sources
    assert "install" in sources
    assert "jax.devices()" in sources
    assert "overfit-one-batch" in sources


def test_notebook_has_no_uv_bootstrap_or_run_commands() -> None:
    """No code cell bootstraps uv or invokes uv run/uv sync."""
    for idx, source in _code_sources():
        assert "uv run" not in source, f"Cell {idx} still uses uv run"
        assert "uv sync" not in source, f"Cell {idx} still uses uv sync"
    nb_text = NOTEBOOK_PATH.read_text()
    assert "~/.cargo/bin/uv" not in nb_text
    assert "~/.local/bin/uv" not in nb_text


def test_notebook_has_gpu_detection_logic() -> None:
    """The notebook detects GPU presence and selects the JAX/viz extras accordingly."""
    sources = "".join(source for _idx, source in _code_sources())
    assert "HAS_GPU" in sources
    assert "cuda12" in sources
    assert "[viz]" in sources


# --- First code cell: user settings only -------------------------------------


def test_first_code_cell_is_settings_only() -> None:
    """The first code cell contains assignments/comments only.

    No helper function definitions, no imports, no subprocess calls, and no
    GPU probing are allowed in the settings cell.
    """
    settings = _settings_source()
    assert "subprocess" not in settings
    assert "def " not in settings
    assert "nvidia-smi" not in settings
    assert "import jax" not in settings
    tree = ast.parse(settings)
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)), (
            "settings cell must not define functions"
        )
        assert not isinstance(node, (ast.Import, ast.ImportFrom)), (
            "settings cell must not import modules"
        )
        assert not isinstance(node, ast.Call), "settings cell must not call functions"


@pytest.mark.parametrize("name", REQUIRED_SETTINGS)
def test_settings_cell_exposes_documented_parameter(name: str) -> None:
    """Each documented user parameter is assigned in the first code cell."""
    settings = _settings_source()
    assert re.search(rf"^{name}\s*=", settings, flags=re.MULTILINE), (
        f"settings cell does not assign {name}"
    )


def test_derived_values_live_in_second_cell() -> None:
    """Derived values are computed in the second cell, not hidden in settings."""
    settings = _settings_source()
    for derived in DERIVED_NAMES:
        assert derived not in settings, f"settings cell must not derive {derived}"
    derivation = _second_code_cell_source()
    for derived in DERIVED_NAMES:
        assert derived in derivation, f"derivation cell must define {derived}"


# --- Resolved execution plan gate ---------------------------------------------


def test_execution_plan_validates_settings_early() -> None:
    """The plan cell rejects bad configs/epochs before any side effect."""
    plan = _plan_cell_source()
    assert ".yaml" in plan
    assert "TRAIN_EPOCHS" in plan
    assert "ValueError" in plan
    # The plan prints every decision a later cell will act on.
    for topic in ("GIT_REF", "TRAIN_PREFIX", "FORCE_FRESH", "ALLOW_DIRTY_REPO", "PIP_EXTRA"):
        assert topic in plan, f"plan does not surface {topic}"


def test_execution_plan_runs_before_repo_sync() -> None:
    """The plan gate is positioned before cloning, installing, or training."""
    plan_idx = _cell_index_containing("Resolved execution plan")
    assert plan_idx < _cell_index_containing("git", "clone", "fetch")
    assert plan_idx < _cell_index_containing("pip", "importlib.metadata")
    assert plan_idx < _first_training_cell_index()


# --- Repository synchronization ------------------------------------------------


def test_repo_sync_updates_existing_clone() -> None:
    """A reused runtime fetches and fast-forward-updates the existing clone."""
    source = _repo_sync_cell_source()
    assert "clone" in source
    assert "fetch" in source
    assert "--ff-only" in source
    assert "rev-parse" in source  # current commit SHA printed after sync


def test_repo_sync_guards_dirty_clone() -> None:
    """Local modifications stop the notebook unless ALLOW_DIRTY_REPO=True."""
    source = _repo_sync_cell_source()
    assert "status" in source
    assert "ALLOW_DIRTY_REPO" in source
    assert "Local modifications" in source


def test_repo_sync_verifies_selected_config() -> None:
    """After synchronization the selected config must exist in the clone."""
    source = _repo_sync_cell_source()
    assert "(REPO / TRAIN_CONFIG).is_file()" in source


# --- Install/repair cell --------------------------------------------------------


def test_notebook_install_cell_uses_pip_with_viz_extra() -> None:
    """The install cell uses pip install -e with the viz extra (and cuda12 on GPU)."""
    source = _install_cell_source()
    assert "pip" in source
    assert "install" in source
    assert "-e" in source
    assert "PIP_EXTRA" in source


def test_notebook_install_cell_never_skips_on_importable_package() -> None:
    """The install must not be skipped merely because the package is importable.

    A reused Colab runtime can hold an incompatible preinstalled JAX/CUDA mix
    that still imports, so the idempotent-skip pattern is forbidden here.
    """
    source = _install_cell_source()
    assert "import spheroid_seg" not in source
    assert "skipping install" not in source


def test_notebook_install_cell_repairs_conflicting_cuda_plugins() -> None:
    """The install cell inspects distributions and repairs conflicting CUDA plugins."""
    source = _install_cell_source()
    assert "importlib.metadata" in source
    assert "REPAIR_JAX_ENVIRONMENT" in source
    assert "uninstall" in source
    # Detection happens via importlib.metadata before JAX is imported.
    assert '"jax" in sys.modules' in source


# --- Post-install JAX sanity gate ------------------------------------------------


def test_notebook_jax_gpu_assertion() -> None:
    """On GPU runtimes the gate raises unless JAX itself reports a GPU device."""
    source = _jax_gate_cell_source()
    assert "import spheroid_seg" in source
    assert "spheroid_seg.__version__" in source
    assert "jax.devices()" in source
    assert "HAS_GPU" in source
    assert "RuntimeError" in source


def test_notebook_has_post_install_sanity_cell() -> None:
    """A sanity cell imports the package and prints JAX devices."""
    sources = "".join(source for _idx, source in _code_sources())
    assert "import spheroid_seg" in sources
    assert "jax.devices()" in sources
    assert "__version__" in sources


def test_jax_sanity_gate_runs_before_any_training() -> None:
    """The GPU gate is positioned before the overfit check and the main training."""
    gate_idx = _cell_index_containing("import jax", "jax.devices()")
    assert gate_idx < _first_training_cell_index()


# --- Optional overfit-one-batch check --------------------------------------------


def test_overfit_cell_guarded_by_user_setting() -> None:
    """The overfit check runs only when HAS_GPU and RUN_OVERFIT_CHECK are both true."""
    source = _overfit_cell_source()
    assert "HAS_GPU" in source
    assert "RUN_OVERFIT_CHECK" in source
    assert "configs/base.yaml" in source
    # When skipped, the cell must print exactly why.
    assert "Skipping overfit-one-batch:" in source


def test_overfit_cell_warns_about_base_run_dir() -> None:
    """The overfit cell explains that it intentionally creates a base_<timestamp> run."""
    source = _overfit_cell_source()
    assert "base_<timestamp>" in source


def test_overfit_cell_is_visually_separated_from_main_training() -> None:
    """The overfit section appears before, and is not merged with, the main training."""
    overfit_idx = _cell_index_containing("overfit-one-batch")
    assert overfit_idx < _drive_resume_cell_index()


# --- Main training cell ------------------------------------------------------------


def test_main_training_cell_uses_resolved_config() -> None:
    """The main training flow uses the resolved config, not a hardcoded one."""
    source = _drive_resume_cell_source()
    assert '"--config"' in source
    assert "train_config" in source
    # No hardcoded config paths outside the optional overfit section.
    for config_path in ("configs/colab.yaml", "configs/base.yaml", "configs/tiny.yaml"):
        assert config_path not in source, f"main training cell hardcodes {config_path}"


def test_main_training_cell_has_no_hidden_epoch_override() -> None:
    """The main training flow must not contain an unconditional epochs = 3 override."""
    source = _drive_resume_cell_source()
    assert "epochs = 3" not in source
    assert "epochs = 2" not in source
    # --epochs is only appended when TRAIN_EPOCHS is an explicit integer.
    assert "TRAIN_EPOCHS is not None" in source
    assert '"--epochs"' in source


def test_notebook_defines_streaming_run_helper() -> None:
    """The notebook defines a streaming run() helper that uses subprocess.Popen."""
    sources = "".join(source for _idx, source in _code_sources())
    assert "def run(" in sources
    assert "subprocess.Popen" in sources


def test_notebook_training_commands_use_run_helper_with_cwd_repo() -> None:
    """Every training subprocess goes through run() with cwd=REPO."""
    for idx, source in _code_sources():
        for node in _run_calls(source):
            if _is_training_run_call(node):
                assert _run_call_has_cwd_repo(node), (
                    f"Cell {idx}: training run() must pass cwd=REPO"
                )


def test_notebook_sets_xla_mem_fraction_for_training() -> None:
    """Every training subprocess sets XLA_PYTHON_CLIENT_MEM_FRACTION=0.9."""
    for idx, source in _code_sources():
        for node in _run_calls(source):
            if _is_training_run_call(node):
                assert _env_dict_has_xla_fraction(node), (
                    f"Cell {idx}: training run() must pass "
                    "env={{'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.9'}}"
                )


def test_notebook_training_cell_prints_exact_command() -> None:
    """The exact resolved subprocess command is printed before launching."""
    source = _drive_resume_cell_source()
    assert "Launching training with command" in source
    assert "train_cmd" in source


def test_notebook_training_cell_scans_drive_for_incomplete_run() -> None:
    """The training cell scans Drive runs for unfinished training-state metadata."""
    source = _drive_resume_cell_source()
    assert "DRIVE_RUNS_DIR" in source
    assert "training_state_metadata.yaml" in source
    assert "FORCE_FRESH" in source
    assert "TRAIN_PREFIX" in source


def test_notebook_training_cell_calls_resume_on_detected_run() -> None:
    """The training cell passes --resume <run dir> for the detected incomplete run."""
    source = _drive_resume_cell_source()
    assert '"--resume"' in source
    assert "str(resume_run_dir)" in source


def test_notebook_training_cell_verifies_drive_mount() -> None:
    """The training cell verifies the Drive mount before using any Drive path."""
    source = _drive_resume_cell_source()
    assert 'os.path.ismount("/content/drive")' in source
    assert "DRIVE_ROOT.exists()" in source


def test_notebook_drive_config_derived_from_config_stem() -> None:
    """The throwaway Drive config path derives from the selected config stem."""
    settings = _settings_source()
    assert "DRIVE_CONFIG_PATH" not in settings, "Drive config path must be derived"
    derivation = _second_code_cell_source()
    assert '"/content/configs"' in derivation
    assert 'f"{CONFIG_STEM}_drive.yaml"' in derivation
    source = _drive_resume_cell_source()
    assert "DRIVE_CONFIG_PATH.parent.mkdir" in source
    assert "DRIVE_CONFIG_PATH.write_text" in source
    # The old fixed /content/colab_drive.yaml location must be gone.
    assert 'Path("/content/colab_drive.yaml")' not in NOTEBOOK_PATH.read_text()


def test_notebook_training_cell_fails_loudly_without_run_dir() -> None:
    """After training, the cell raises unless a matching run directory exists."""
    source = _drive_resume_cell_source()
    assert "RUN_DIR" in source
    assert "best_checkpoint.msgpack" in source
    assert "train_log.csv" in source
    assert "patch_class_prevalence.csv" in source
    assert "RuntimeError" in source


def test_notebook_local_mode_warns_about_ephemeral_checkpoints() -> None:
    """Without a verified Drive mount the training cell warns about ephemeral outputs."""
    source = _drive_resume_cell_source()
    assert "EPHEMERAL" in source


def test_notebook_defines_force_fresh_flag() -> None:
    """FORCE_FRESH is defined at the top of the notebook with a default of False."""
    setup_source = _settings_source()
    assert "FORCE_FRESH = False" in setup_source


def test_notebook_has_no_pytest_cell() -> None:
    """No code cell runs the pytest suite anymore."""
    for idx, source in _code_sources():
        assert "pytest" not in source, f"Cell {idx} still references pytest"


# --- Drive data cell -----------------------------------------------------------------


def test_notebook_has_drive_data_cell_before_training() -> None:
    """The Drive-loading cell exists and is positioned before any training cell."""
    drive_idx = _drive_code_cell_index()
    train_idx = _first_training_cell_index()
    assert drive_idx < train_idx, (
        f"Drive cell ({drive_idx}) must appear before first training cell ({train_idx})"
    )


def test_notebook_drive_cell_is_gated_by_flag() -> None:
    """The Drive-loading cell is gated by a USE_DRIVE_DATA-style flag."""
    idx = _drive_code_cell_index()
    source = dict(_code_sources())[idx]
    assert "USE_DRIVE_DATA" in source
    # Flag is defined at the top of the notebook with a default of False.
    setup_source = _settings_source()
    assert "USE_DRIVE_DATA = False" in setup_source


def test_notebook_drive_cell_uses_copytree() -> None:
    """The Drive-loading cell copies data with shutil.copytree(..., dirs_exist_ok=True)."""
    idx = _drive_code_cell_index()
    source = dict(_code_sources())[idx]
    assert "shutil.copytree" in source
    assert "dirs_exist_ok=True" in source


def test_notebook_drive_cell_prints_sanity_check() -> None:
    """The Drive-loading cell prints file counts and warns when data is missing."""
    idx = _drive_code_cell_index()
    source = dict(_code_sources())[idx]
    assert 'REPO / "data" / "raw"' in source or "REPO / 'data' / 'raw'" in source
    assert 'REPO / "data" / "masks"' in source or "REPO / 'data' / 'masks'" in source
    assert "WARNING" in source
    assert "DRIVE_DATA_DIR" in source


def test_notebook_drive_cell_fails_when_drive_paths_missing() -> None:
    """Enabled Drive data with a missing Drive path raises instead of falling back."""
    idx = _drive_code_cell_index()
    source = dict(_code_sources())[idx]
    assert "FileNotFoundError" in source


def test_notebook_only_drive_data_cell_mounts_drive() -> None:
    """drive.mount appears in exactly one cell: the Drive data-loading cell."""
    mount_cells = [idx for idx, source in _code_sources() if "drive.mount" in source]
    assert mount_cells == [_drive_code_cell_index()]


def test_notebook_no_shell_command_uses_drive_paths() -> None:
    """No shell invocation passes Drive paths through a shell string."""
    drive_path_markers = ("/content/drive", "MyDrive", "Colab Notebooks")
    for idx, source in _code_sources():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Reject os.system outright.
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "os"
                    and func.attr == "system"
                ):
                    raise AssertionError(f"Cell {idx}: os.system is not allowed")

                # Reject subprocess.run/Popen with a string command containing Drive markers.
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "subprocess"
                    and func.attr in ("run", "Popen")
                ):
                    first_arg = node.args[0] if node.args else None
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        cmd = first_arg.value
                        if any(marker in cmd for marker in drive_path_markers):
                            raise AssertionError(
                                f"Cell {idx}: subprocess.{func.attr} receives a Drive path string"
                            )


# --- Training curves and persistence cells ----------------------------------------------


def test_notebook_curves_cell_uses_resolved_run_dir() -> None:
    """The training-curves cell reads the run dir located by the training cell."""
    curves_idx = _cell_index_containing("train_log.csv", "matplotlib")
    source = dict(_code_sources())[curves_idx]
    assert "RUN_DIR" in source
    assert "train_log.csv" in source
    assert "matplotlib.pyplot" in source


def test_notebook_plot_cell_reads_train_log_csv() -> None:
    """The training-curves cell reads the per-run train_log.csv."""
    sources = "".join(source for _idx, source in _code_sources())
    assert "train_log.csv" in sources
    assert "matplotlib.pyplot" in sources


def test_notebook_download_cell_handles_drive_outputs() -> None:
    """The zip/download cell prints the Drive path instead of archiving when on Drive."""
    dl_idx = next(idx for idx, source in _code_sources() if "files.download" in source)
    source = dict(_code_sources())[dl_idx]
    assert "DRIVE_RUNS_DIR" in source


def test_notebook_documents_drive_resume() -> None:
    """Notebook markdown explains the session-cut resume behavior and FORCE_FRESH."""
    text = "\n".join(_markdown_sources())
    assert "FORCE_FRESH" in text
    assert "resume" in text.lower()


def test_notebook_documents_controlled_experiment_settings() -> None:
    """Notebook markdown documents the controlled-experiment configuration."""
    text = "\n".join(_markdown_sources())
    assert "colab_bgweight05" in text
    assert "RUN_OVERFIT_CHECK = False" in text
    assert "TRAIN_EPOCHS = None" in text
    assert "base_<timestamp>" in text


# --- Subprocess discipline ----------------------------------------------------------------


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


def _call_is_read_only_git(node: ast.Call) -> bool:
    """Return True if the subprocess.run call is a read-only git inspection.

    Matches `git [-C <repo>] <subcommand> ...` where <subcommand> is in the
    read-only whitelist. The repo path may be a string literal, a str() call, or
    an f-string; those tokens are opaque here because the whitelist check only
    inspects the subcommand.
    """
    first_arg = node.args[0] if node.args else None
    if not isinstance(first_arg, ast.List) or not first_arg.elts:
        return False
    tokens: list[tuple[str, bool]] = []
    for elt in first_arg.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            tokens.append((elt.value, False))
        elif isinstance(elt, (ast.Call, ast.JoinedStr, ast.Name)):
            tokens.append(("", True))  # opaque non-literal argument
        else:
            return False
    if not tokens or tokens[0] != ("git", False):
        return False
    rest = tokens[1:]
    if rest and rest[0] == ("-C", False):
        rest = rest[2:]  # skip "-C" plus its path argument (literal or opaque)
    return bool(rest) and not rest[0][1] and rest[0][0] in READ_ONLY_GIT_SUBCOMMANDS


def test_notebook_command_cells_use_streaming_helper() -> None:
    """Command invocations go through the streaming run() helper.

    The only allowed bare subprocess.run calls are the one-off GPU detection
    probe and read-only git inspection (rev-parse/status/diff/...); every
    mutating command (clone, fetch, checkout, merge, pip, zip, training) must
    use the helper so its output streams live.
    """
    for idx, source in _code_sources():
        for node in _subprocess_run_calls(source):
            assert _call_targets_nvidia_smi(node) or _call_is_read_only_git(node), (
                f"Cell {idx}: bare subprocess.run call found; "
                "route repo commands through the run() helper"
            )


def _run_calls(source: str) -> list[ast.Call]:
    """Return all calls to the notebook's run() helper."""
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run":
            calls.append(node)
    return calls


def _is_training_run_call(node: ast.Call) -> bool:
    """Return True if the run() call invokes spheroid_seg.train.

    The command is either an inline list containing "spheroid_seg.train", or a
    variable holding that list: the main training cell builds its command in a
    variable so the exact resolved command can be printed before launching.
    """
    first_arg = node.args[0] if node.args else None
    if isinstance(first_arg, ast.Name):
        return True
    if not isinstance(first_arg, ast.List):
        return False
    parts = [
        elt.value
        for elt in first_arg.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    ]
    return "spheroid_seg.train" in parts


def _run_call_has_cwd_repo(node: ast.Call) -> bool:
    """Return True if the run() call passes cwd=REPO."""
    for kw in node.keywords:
        if (
            isinstance(kw, ast.keyword)
            and kw.arg == "cwd"
            and isinstance(kw.value, ast.Name)
            and kw.value.id == "REPO"
        ):
            return True
    return False


def _env_dict_has_xla_fraction(node: ast.Call) -> bool:
    """Return True if the run() call passes XLA_PYTHON_CLIENT_MEM_FRACTION=0.9."""
    for kw in node.keywords:
        if kw.arg == "env" and isinstance(kw.value, ast.Dict):
            for key, value in zip(kw.value.keys, kw.value.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "XLA_PYTHON_CLIENT_MEM_FRACTION"
                    and isinstance(value, ast.Constant)
                    and value.value == "0.9"
                ):
                    return True
    return False


def test_configs_colab_yaml_matches_base() -> None:
    """configs/colab.yaml equals base.yaml except for batch_size, which is 4."""
    import yaml

    base_path = REPO_ROOT / "configs" / "base.yaml"
    colab_path = REPO_ROOT / "configs" / "colab.yaml"
    assert colab_path.is_file(), "configs/colab.yaml does not exist"

    base_cfg = yaml.safe_load(base_path.read_text())
    colab_cfg = yaml.safe_load(colab_path.read_text())

    assert colab_cfg.get("batch_size") == 4, "colab.yaml batch_size must be 4"

    # Compare all keys except batch_size.
    base_copy = {k: v for k, v in base_cfg.items() if k != "batch_size"}
    colab_copy = {k: v for k, v in colab_cfg.items() if k != "batch_size"}
    assert base_copy == colab_copy, "colab.yaml differs from base.yaml beyond batch_size"
