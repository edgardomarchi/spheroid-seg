"""Flax U-Net for semantic segmentation."""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


def default_config() -> dict[str, Any]:
    """Return the default U-Net configuration matching configs/base.yaml."""
    return {
        "num_classes": 3,
        "base_features": 32,
        "input_channels": "grayscale",
    }


def count_parameters(params: Any) -> int:
    """Count the total number of scalar parameters in a parameter pytree."""
    return int(sum(p.size for p in jax.tree_util.tree_leaves(params)))


class ConvBlock(nn.Module):
    """Two Conv 3x3 (same padding) → BatchNorm → ReLU blocks."""

    features: int
    train: bool
    bn_momentum: float = 0.99

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the convolutional block."""
        for _ in range(2):
            x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.BatchNorm(use_running_average=not self.train, momentum=self.bn_momentum)(x)
            x = nn.relu(x)
        return x


class UNet(nn.Module):
    """U-Net from scratch in Flax.

    Attributes:
        num_classes: Number of output logits per pixel.
        base_features: Number of features in the first encoder level.
        input_channels: Expected input channel layout (used only for validation).
    """

    num_classes: int = 3
    base_features: int = 32
    input_channels: str = "grayscale"
    bn_momentum: float = 0.99

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass returning per-pixel logits.

        Args:
            x: Input tensor of shape (B, H, W, C).
            train: Whether to update batch-norm statistics.

        Returns:
            Logits tensor of shape (B, H, W, num_classes).

        Raises:
            ValueError: If the input spatial size is not divisible by 16.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,H,W,C), got shape {x.shape}")

        _, height, width, _ = x.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Input spatial size ({height}, {width}) must be divisible by 16 "
                "for the 4-level U-Net encoder/decoder."
            )

        expected_channels = 1 if self.input_channels == "grayscale" else 3
        if x.shape[-1] != expected_channels:
            raise ValueError(
                f"input_channels={self.input_channels!r} expects {expected_channels} "
                f"channels, got {x.shape[-1]}"
            )

        # Encoder
        e0 = ConvBlock(features=self.base_features, train=train, bn_momentum=self.bn_momentum)(x)
        e1_in = nn.max_pool(e0, window_shape=(2, 2), strides=(2, 2))

        e1 = ConvBlock(features=self.base_features * 2, train=train, bn_momentum=self.bn_momentum)(
            e1_in
        )
        e2_in = nn.max_pool(e1, window_shape=(2, 2), strides=(2, 2))

        e2 = ConvBlock(features=self.base_features * 4, train=train, bn_momentum=self.bn_momentum)(
            e2_in
        )
        e3_in = nn.max_pool(e2, window_shape=(2, 2), strides=(2, 2))

        e3 = ConvBlock(features=self.base_features * 8, train=train, bn_momentum=self.bn_momentum)(
            e3_in
        )
        e4_in = nn.max_pool(e3, window_shape=(2, 2), strides=(2, 2))

        # Bottleneck
        bottleneck = ConvBlock(
            features=self.base_features * 16, train=train, bn_momentum=self.bn_momentum
        )(e4_in)

        # Decoder
        d3 = nn.ConvTranspose(
            features=self.base_features * 8, kernel_size=(2, 2), strides=(2, 2), padding="SAME"
        )(bottleneck)
        d3 = jnp.concatenate([d3, e3], axis=-1)
        d3 = ConvBlock(features=self.base_features * 8, train=train, bn_momentum=self.bn_momentum)(
            d3
        )

        d2 = nn.ConvTranspose(
            features=self.base_features * 4, kernel_size=(2, 2), strides=(2, 2), padding="SAME"
        )(d3)
        d2 = jnp.concatenate([d2, e2], axis=-1)
        d2 = ConvBlock(features=self.base_features * 4, train=train, bn_momentum=self.bn_momentum)(
            d2
        )

        d1 = nn.ConvTranspose(
            features=self.base_features * 2, kernel_size=(2, 2), strides=(2, 2), padding="SAME"
        )(d2)
        d1 = jnp.concatenate([d1, e1], axis=-1)
        d1 = ConvBlock(features=self.base_features * 2, train=train, bn_momentum=self.bn_momentum)(
            d1
        )

        d0 = nn.ConvTranspose(
            features=self.base_features, kernel_size=(2, 2), strides=(2, 2), padding="SAME"
        )(d1)
        d0 = jnp.concatenate([d0, e0], axis=-1)
        d0 = ConvBlock(features=self.base_features, train=train, bn_momentum=self.bn_momentum)(d0)

        # Head
        logits = nn.Conv(features=self.num_classes, kernel_size=(1, 1), padding="SAME")(d0)
        return logits
