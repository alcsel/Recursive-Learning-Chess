"""Shared constants and runtime configuration.

This module deliberately imports nothing heavy: it backs the command line
interface and the unit tests that run without JAX installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BLACK",
    "COLOUR_NAMES",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_FROZEN_UPDATE_INTERVAL",
    "DEFAULT_NUM_ENVS",
    "DEFAULT_SEED",
    "WHITE",
    "PlayConfig",
    "TrainingConfig",
    "colour_name",
    "epsilon_for",
]

WHITE = 0
BLACK = 1
COLOUR_NAMES = {WHITE: "White", BLACK: "Black"}

DEFAULT_NUM_ENVS = 2048
DEFAULT_FROZEN_UPDATE_INTERVAL = 500
DEFAULT_SEED = 42
DEFAULT_CHECKPOINT = Path("checkpoints") / "chess_brain.msgpack"

_EPSILON_SCHEDULE = ((1000, 0.5), (3000, 0.2), (float("inf"), 0.05))


def colour_name(colour: int) -> str:
    """Return the human readable name of a colour."""
    return COLOUR_NAMES.get(colour, f"colour {colour}")


def epsilon_for(batch: int, schedule: tuple[tuple[int, float], ...] = _EPSILON_SCHEDULE) -> float:
    """Return the exploration rate used for ``batch``.

    ``schedule`` holds ``(upper_bound, epsilon)`` pairs in ascending order; the
    first pair whose bound has not been reached wins, otherwise the rate of the
    last pair is used for the rest of the run.
    """
    if batch < 1:
        raise ValueError("batch must be a positive integer")

    for upper_bound, epsilon in schedule:
        if batch < upper_bound:
            return epsilon
    return schedule[-1][1] if schedule else 0.05


@dataclass(frozen=True)
class TrainingConfig:
    """Everything the training loop needs, with sensible local defaults."""

    num_envs: int = DEFAULT_NUM_ENVS
    batches: int = 100_000
    seed: int = DEFAULT_SEED
    gamma: float = 0.99
    learning_rate: float = 1e-4
    clip_by_global_norm: float = 1.0
    entropy_coefficient: float = 0.01
    frozen_update_interval: int = DEFAULT_FROZEN_UPDATE_INTERVAL
    save_every: int = 25
    log_every: int = 10
    checkpoint: Path = DEFAULT_CHECKPOINT
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if self.batches < 1:
            raise ValueError("batches must be at least 1")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie between 0 and 1")
        if self.frozen_update_interval < 1:
            raise ValueError("frozen_update_interval must be at least 1")
        if self.save_every < 1 or self.log_every < 1:
            raise ValueError("save_every and log_every must be at least 1")

    def should_update_opponent(self, batch: int) -> bool:
        return batch % self.frozen_update_interval == 0

    def should_save(self, batch: int) -> bool:
        return batch % self.save_every == 0

    def should_log(self, batch: int) -> bool:
        return batch % self.log_every == 0


@dataclass(frozen=True)
class PlayConfig:
    """Settings for a single human versus model game."""

    checkpoint: Path = DEFAULT_CHECKPOINT
    model_colour: int = WHITE
    seed: int = DEFAULT_SEED
    show_board: bool = True

    def __post_init__(self) -> None:
        if self.model_colour not in (WHITE, BLACK):
            raise ValueError("model_colour must be 0 (white) or 1 (black)")

    @property
    def human_colour(self) -> int:
        return BLACK if self.model_colour == WHITE else WHITE
