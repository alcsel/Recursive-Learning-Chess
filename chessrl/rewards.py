"""Reward shaping and bootstrapping targets.

PGX always writes the observation from the point of view of the player to move:
planes 0-5 are that player's pieces and planes 6-11 the opponent's. Every term
below is therefore expressed for the player who just moved, and the reward for
a capture is positive.

The formula matches the one documented in the README, with one correction: the
original compared a position against the *opponent's* view of the next
position, which both inverted the sign for half of the parallel games (PGX
assigns colours to player slots at random) and doubled the value. Both sides of
the comparison are now measured from the mover's own point of view.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "DRAW_PENALTY",
    "MATERIAL_SCALE",
    "MATERIAL_WEIGHT",
    "MOBILITY_WEIGHT",
    "PIECE_VALUES",
    "STEP_PENALTY",
    "STALEMATE_MATERIAL_MARGIN",
    "STALEMATE_PENALTY",
    "TD_CLIP",
    "material_after_own_move",
    "material_balance",
    "material_gain",
    "shaped_reward",
    "td_target",
]

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1.0, 3.0, 3.0, 5.0, 9.0, 0.0
PIECE_VALUES = jnp.array([PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING], dtype=jnp.float32)

MATERIAL_SCALE = 39.0
MATERIAL_WEIGHT = 0.3
MOBILITY_WEIGHT = 0.2
STALEMATE_PENALTY = -1.0
DRAW_PENALTY = -0.5
STEP_PENALTY = -0.001
STALEMATE_MATERIAL_MARGIN = 5.0
TD_CLIP = 5.0


def material_balance(observation):
    """Return the material of the player to move minus the opponent's.

    Works on a batch: the result has one entry per environment.
    """
    own = observation[:, :, :, 0:6]
    other = observation[:, :, :, 6:12]
    return jnp.sum(own * PIECE_VALUES, axis=(1, 2, 3)) - jnp.sum(
        other * PIECE_VALUES, axis=(1, 2, 3)
    )


def material_after_own_move(next_observation):
    """Return the mover's material once their move has been applied.

    ``next_observation`` belongs to the opponent, who is now to move, so the
    balance comes back with its sign flipped.
    """
    return -material_balance(next_observation)


def material_gain(before, after):
    """Bonus for the material the mover gained with the move just played.

    ``before`` and ``after`` are both the mover's own balance, so capturing a
    queen yields ``+9 / 39`` before clipping. ``MATERIAL_SCALE`` is the value of
    one full army (39), which is why the result lands in ``[-1, 1]``.
    """
    return jnp.clip((after - before) / MATERIAL_SCALE, -1.0, 1.0) * MATERIAL_WEIGHT


def shaped_reward(
    game_reward,
    gain,
    enemy_mobility,
    terminated,
    current_balance,
    num_actions,
):
    """Combine the terminal result with the shaping terms from the README.

    ``current_balance`` is the mover's material before the move, used to decide
    whether a draw threw away a winning position.
    """
    mobility_penalty = -(enemy_mobility / num_actions) * MOBILITY_WEIGHT

    is_draw = terminated & (game_reward == 0.0)
    is_stalemate_trap = is_draw & (current_balance > STALEMATE_MATERIAL_MARGIN)
    draw_penalty = jnp.where(is_stalemate_trap, STALEMATE_PENALTY, 0.0)
    draw_penalty = draw_penalty + jnp.where(is_draw & ~is_stalemate_trap, DRAW_PENALTY, 0.0)

    return game_reward + gain + mobility_penalty + draw_penalty + STEP_PENALTY


def td_target(reward, next_value, terminated, gamma=0.99, clip=TD_CLIP):
    """Bootstrap one step ahead and keep the target inside a sane range."""
    bootstrap = gamma * next_value * (1.0 - terminated.astype(jnp.float32))
    return jnp.clip(reward + bootstrap, -clip, clip)
