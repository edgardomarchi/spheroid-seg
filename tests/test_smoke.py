"""Smoke tests for package scaffolding, config loading, and CLI stubs."""

import subprocess
import sys
from pathlib import Path

import yaml

import spheroid_seg
from spheroid_seg import data, models, postproc

REQUIRED_CONFIG_KEYS = {
    "num_classes",
    "patch_size",
    "base_features",
    "lr",
    "weight_decay",
    "batch_size",
    "epochs",
    "early_stopping_patience",
    "seed",
    "data",
    "outputs",
    "input_channels",
    "class_mapping",
    "min_object_fraction",
    "object_patch_ratio",
    "patches_per_image",
    "class_weights",
    "augment",
}


def test_version() -> None:
    """The package exposes a version string."""
    assert isinstance(spheroid_seg.__version__, str)
    assert len(spheroid_seg.__version__) > 0


def test_subpackages_import() -> None:
    """Core subpackages are importable."""
    assert data is not None
    assert models is not None
    assert postproc is not None


def test_config_loads() -> None:
    """The base config exists and contains the required keys."""
    config_path = Path("configs/base.yaml")
    assert config_path.exists()

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    assert REQUIRED_CONFIG_KEYS.issubset(config.keys())
    assert config["num_classes"] == 3
    assert config["patch_size"] == 512
    assert config["base_features"] == 32
    assert config["lr"] == 1.0e-3
    assert config["weight_decay"] == 1.0e-4
    assert config["batch_size"] == 8


def _cli_help_exits_zero(module: str) -> None:
    """Run a module's CLI with --help and assert a clean exit."""
    result = subprocess.run(
        [sys.executable, "-m", f"spheroid_seg.{module}", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_train_help() -> None:
    """The train CLI prints help and exits 0."""
    _cli_help_exits_zero("train")


def test_eval_help() -> None:
    """The eval CLI prints help and exits 0."""
    _cli_help_exits_zero("eval")


def test_infer_help() -> None:
    """The infer CLI prints help and exits 0."""
    _cli_help_exits_zero("infer")
