"""Tests for the training loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from spheroid_seg.train import create_train_state, train_step


def _fixed_batch(
    rng: np.random.Generator,
    batch_size: int = 2,
    size: int = 128,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create a fixed synthetic batch with a few labeled circles."""
    images = rng.random((batch_size, size, size, 1)).astype(np.float32)
    masks = np.zeros((batch_size, size, size), dtype=np.int32)
    for b in range(batch_size):
        masks[b, size // 4 : size // 2, size // 4 : size // 2] = 1
        masks[b, size // 2 : 3 * size // 4, size // 2 : 3 * size // 4] = 2
    return jnp.array(images), jnp.array(masks)


def test_train_step_reduces_loss() -> None:
    """A single training step reduces the loss on a fixed batch."""
    rng = np.random.default_rng(42)
    config = {
        "num_classes": 3,
        "base_features": 8,
        "input_channels": "grayscale",
        "patch_size": 128,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "epochs": 1,
        "batch_size": 2,
        "class_weights": [0.1, 1.0, 1.0],
        "seed": 0,
    }

    batch = _fixed_batch(rng, batch_size=config["batch_size"], size=config["patch_size"])
    class_weights = jnp.array(config["class_weights"], dtype=jnp.float32)

    state = create_train_state(config, jax.random.PRNGKey(config["seed"]), steps_per_epoch=1)

    losses: list[float] = []
    for _ in range(5):
        state, step_loss = train_step(state, batch, class_weights)
        losses.append(float(step_loss))

    assert losses[-1] < losses[0]
