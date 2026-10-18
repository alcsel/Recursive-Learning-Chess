"""The residual convolutional network that plays chess."""

from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn

__all__ = ["NUM_ACTIONS", "OBSERVATION_SHAPE", "ChessBrain", "ResidualBlock"]

NUM_ACTIONS = 4672
OBSERVATION_SHAPE = (8, 8, 119)
DEFAULT_NUM_BLOCKS = 8
DEFAULT_FEATURES = 128


class ResidualBlock(nn.Module):
    """Two convolutions with a skip connection.

    ``Conv -> LayerNorm -> ReLU -> Conv -> LayerNorm -> (+x) -> ReLU``
    """

    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = x + residual
        return nn.relu(x)


class ChessBrain(nn.Module):
    """Policy and value head over the PGX chess observation.

    Takes an ``(N, 8, 8, 119)`` observation and an ``(N, 4672)`` legal action
    mask and returns masked policy logits plus a scalar value per position.
    """

    num_blocks: int = DEFAULT_NUM_BLOCKS
    features: int = DEFAULT_FEATURES

    @nn.compact
    def __call__(self, observation, legal_action_mask):
        x = observation.astype(jnp.float32)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        for _ in range(self.num_blocks):
            x = ResidualBlock(features=self.features)(x)

        x = x.reshape((x.shape[0], -1))
        logits = nn.Dense(features=NUM_ACTIONS)(x)
        logits = jnp.where(legal_action_mask, logits, -1e9)
        value = nn.Dense(features=1)(x)
        return logits, value
