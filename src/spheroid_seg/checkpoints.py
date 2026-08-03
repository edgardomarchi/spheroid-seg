"""Shared checkpoint resolution utilities for train/eval/infer."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_checkpoint(
    config: dict[str, Any],
    config_path: Path | str,
    run_dir: Path | str | None = None,
    checkpoint: Path | str | None = None,
) -> Path:
    """Resolve the checkpoint path using the precedence: explicit > run-dir > latest.

    Args:
        config: Loaded configuration dictionary (used to locate the runs dir).
        config_path: Path to the config file (used to derive the config stem).
        run_dir: Optional run directory containing ``checkpoints/best_checkpoint.msgpack``.
        checkpoint: Optional explicit checkpoint file path.

    Returns:
        Path to the resolved checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint can be resolved.
    """
    if checkpoint is not None:
        path = Path(checkpoint)
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if run_dir is not None:
        path = Path(run_dir) / "checkpoints" / "best_checkpoint.msgpack"
        if path.exists():
            return path
        raise FileNotFoundError(f"No best checkpoint found in run directory: {run_dir}")

    config_stem = Path(config_path).stem
    runs_dir = Path(config["outputs"]["checkpoints_dir"]).parent / "runs"
    candidates = sorted(
        runs_dir.glob(f"{config_stem}_*/checkpoints/best_checkpoint.msgpack"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint resolved for config '{config_stem}'. "
        "Provide --checkpoint or --run-dir, or ensure outputs/runs/<config>_* exists."
    )
