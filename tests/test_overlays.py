"""Tests for overlay grid generation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from spheroid_seg.overlays import CLASS_COLORMAP, build_overlay_grid


def _make_sample(name: str, size: tuple[int, int] = (64, 64)) -> dict:
    """Create a tiny deterministic sample for overlay tests."""
    rng = np.random.default_rng(hash(name) % 2**32)
    raw = rng.integers(0, 255, size=size, dtype=np.uint8)
    gt = np.zeros(size, dtype=np.uint8)
    gt[size[0] // 4 : size[0] // 2, size[1] // 4 : size[1] // 2] = 1
    gt[size[0] // 2 : 3 * size[0] // 4, size[1] // 2 : 3 * size[1] // 4] = 2
    pred = gt.copy()
    return {
        "name": name,
        "magnification": "4x" if "_4x" in name else "10x",
        "raw": raw,
        "gt": gt,
        "pred": pred,
    }


def test_overlay_grid_shape() -> None:
    """The grid has rows x 4 columns and the requested panel width."""
    samples = [_make_sample(f"synth_{i:03d}_4x.png") for i in range(2)]
    samples += [_make_sample(f"synth_{i:03d}_10x.png") for i in range(2, 4)]

    grid = build_overlay_grid(samples, panel_width=128)
    assert grid.shape[0] == len(samples) * 128
    assert grid.shape[1] == 4 * 128
    assert grid.shape[2] == 3


def test_error_overlay_shows_false_positive_red() -> None:
    """A forced false positive produces red pixels in the error column."""
    sample = _make_sample("synth_000_4x.png", size=(32, 32))
    sample["pred"][0, 0] = 1  # false positive for class 1

    grid = build_overlay_grid([sample], panel_width=64)
    # Last quarter of the row is the error overlay.
    error_panel = grid[:, 3 * 64 : 4 * 64]
    red = np.array([0, 0, 255], dtype=np.uint8)
    assert np.any(np.all(error_panel == red, axis=-1))


def test_error_overlay_shows_false_negative_blue() -> None:
    """A forced false negative produces blue pixels in the error column."""
    sample = _make_sample("synth_000_4x.png", size=(32, 32))
    sample["pred"][sample["gt"] == 1] = 0  # false negative for class 1

    grid = build_overlay_grid([sample], panel_width=64)
    error_panel = grid[:, 3 * 64 : 4 * 64]
    blue = np.array([255, 0, 0], dtype=np.uint8)
    assert np.any(np.all(error_panel == blue, axis=-1))


def test_overlay_grid_deterministic() -> None:
    """Identical inputs produce byte-identical encoded PNG grids."""
    samples = [_make_sample(f"synth_{i:03d}_4x.png") for i in range(2)]

    grid_a = build_overlay_grid(samples, panel_width=64)
    grid_b = build_overlay_grid(samples, panel_width=64)

    np.testing.assert_array_equal(grid_a, grid_b)

    _, enc_a = cv2.imencode(".png", grid_a)
    _, enc_b = cv2.imencode(".png", grid_b)
    np.testing.assert_array_equal(enc_a, enc_b)


def test_class_colormap_documented() -> None:
    """The colormap covers the three model classes."""
    assert set(CLASS_COLORMAP.keys()) == {0, 1, 2}
    for color in CLASS_COLORMAP.values():
        assert len(color) == 3


def _load_visualize_batches():
    script = Path(__file__).resolve().parent.parent / "scripts" / "visualize_batches.py"
    spec = importlib.util.spec_from_file_location("visualize_batches", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overlay_grayscale_float_image_is_visible():
    """Regression: 2D float [0,1] images must be scaled to uint8, not truncated."""
    _overlay = _load_visualize_batches()._overlay
    image = np.full((64, 64), 0.6, dtype=np.float32)
    mask = np.zeros((64, 64), dtype=np.uint8)
    out = _overlay(image, mask)
    assert out.dtype == np.uint8
    assert out[0, 0].mean() > 50  # background renders near 0.6*255, not 0


class _MatplotlibBlocker:
    """Meta-path finder that makes matplotlib (and its submodules) unimportable."""

    def find_spec(self, name: str, path: object = None, target: object = None) -> object:
        if name == "matplotlib" or name.startswith("matplotlib."):
            spec = importlib.util.spec_from_loader(name, self)
            return spec
        return None

    def create_module(self, spec: object) -> None:
        return None

    def exec_module(self, module: object) -> None:
        raise ModuleNotFoundError(f"No module named {module.__name__!r}")


def test_visualize_batches_requires_matplotlib_for_plotting() -> None:
    """The plotting path raises a clear error when matplotlib is unavailable."""
    real_modules = sys.modules.copy()
    for name in list(sys.modules):
        if name.startswith("matplotlib"):
            del sys.modules[name]

    blocker = _MatplotlibBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        module = _load_visualize_batches()
        with pytest.raises(ModuleNotFoundError, match="uv sync --all-groups"):
            module._get_pyplot()
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.clear()
        sys.modules.update(real_modules)
