"""Tests for the Flax U-Net model (module M2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from spheroid_seg.models.unet import UNet, count_parameters, default_config


@pytest.fixture
def rng() -> jax.Array:
    """Reproducible PRNG key for parameter initialization."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def default_model() -> UNet:
    """Default U-Net matching configs/base.yaml defaults."""
    return UNet(num_classes=3, base_features=32, input_channels="grayscale")


def _dummy_input(batch_size: int, size: int, channels: int) -> jnp.ndarray:
    """Create a dummy input batch."""
    return jnp.ones((batch_size, size, size, channels), dtype=jnp.float32)


def _tree_allclose(a: object, b: object) -> bool:
    """Return True if two pytrees of arrays are all-close."""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    return all(
        jnp.allclose(lea, leb) for lea, leb in zip(leaves_a, leaves_b, strict=False)
    )


@pytest.mark.parametrize("input_channels", ["grayscale", "rgb"])
def test_output_shape_512(rng: jax.Array, input_channels: str) -> None:
    """A 512x512 input produces logits with the same spatial size and num_classes."""
    model = UNet(num_classes=3, base_features=32, input_channels=input_channels)
    channels = 1 if input_channels == "grayscale" else 3
    x = _dummy_input(1, 512, channels)
    params = model.init(rng, x)
    logits = model.apply(params, x)

    assert logits.shape == (1, 512, 512, 3)
    assert jnp.issubdtype(logits.dtype, jnp.floating)


def test_output_shape_256(default_model: UNet, rng: jax.Array) -> None:
    """A 256x256 grayscale input also yields correctly sized logits."""
    x = _dummy_input(1, 256, 1)
    params = default_model.init(rng, x)
    logits = default_model.apply(params, x)

    assert logits.shape == (1, 256, 256, 3)


def test_batch_size_greater_than_one(default_model: UNet, rng: jax.Array) -> None:
    """The model handles batches larger than one."""
    x = _dummy_input(4, 512, 1)
    params = default_model.init(rng, x)
    logits = default_model.apply(params, x)

    assert logits.shape == (4, 512, 512, 3)


def test_determinism(default_model: UNet, rng: jax.Array) -> None:
    """Initializing with the same RNG twice gives identical parameters."""
    x = _dummy_input(1, 512, 1)
    params_a = default_model.init(rng, x)
    params_b = default_model.init(rng, x)

    assert _tree_allclose(params_a, params_b)


def test_different_rng_gives_different_params(default_model: UNet, rng: jax.Array) -> None:
    """Initializing with different RNGs gives different parameters."""
    x = _dummy_input(1, 512, 1)
    params_a = default_model.init(rng, x)
    _, rng_b = jax.random.split(rng)
    params_b = default_model.init(rng_b, x)

    assert not _tree_allclose(params_a, params_b)


def test_gradient_flow(default_model: UNet, rng: jax.Array) -> None:
    """Gradients w.r.t. all parameters are finite for a simple scalar loss."""
    x = _dummy_input(1, 512, 1)
    params = default_model.init(rng, x)

    def loss_fn(p: dict) -> jnp.ndarray:
        logits = default_model.apply(p, x)
        return jnp.mean(logits**2)

    grads = jax.grad(loss_fn)(params)

    flat_grads = jax.tree_util.tree_leaves_with_path(grads)
    for path, grad in flat_grads:
        name = "/".join(str(k) for k in path)
        assert jnp.isfinite(grad).all(), f"non-finite gradient at {name}"


def test_rejects_non_divisible_input_size(default_model: UNet, rng: jax.Array) -> None:
    """Input sizes not divisible by 2^4 are rejected with a clear error."""
    x = _dummy_input(1, 500, 1)

    with pytest.raises(ValueError, match="divisible by 16"):
        default_model.init(rng, x)


def test_default_config_matches_base_yaml() -> None:
    """The default_config helper reflects the project's default YAML values."""
    cfg = default_config()
    assert cfg["num_classes"] == 3
    assert cfg["base_features"] == 32
    assert cfg["input_channels"] == "grayscale"


def test_parameter_count_reported(default_model: UNet, rng: jax.Array) -> None:
    """The parameter-count utility returns a positive finite integer."""
    x = _dummy_input(1, 512, 1)
    params = default_model.init(rng, x)
    n_params = count_parameters(params)

    assert isinstance(n_params, int)
    assert n_params > 0
