"""PGX chess environment access and UCI translation.

Two conventions from PGX matter here:

``_player_order``
    ``pgx.make("chess")`` randomises, once per environment, which *player slot*
    controls which colour. ``state._player_order[colour]`` therefore returns the
    player id for ``colour`` (0 = white, 1 = black) and must never be assumed to
    be ``[0, 1]``.

``observation``
    The observation is always written from the point of view of the player to
    move: planes 0-5 are the mover's pieces, planes 6-11 the opponent's. It is
    not an absolute white/black board.
"""

from __future__ import annotations

import chess
import jax
import jax.numpy as jnp
import pgx
from pgx.experimental.chess import to_fen

from .config import BLACK, WHITE
from .model import NUM_ACTIONS

__all__ = [
    "BLACK",
    "WHITE",
    "action_to_uci",
    "board_of",
    "env",
    "legal_action_indices",
    "legal_ucis",
    "player_id_for_colour",
    "reset_environments",
    "reset_terminated",
    "step_environment",
    "step_environments",
    "uci_to_action",
]

env = pgx.make("chess")

# ``pgx.core.Env.step`` executes the whole step eagerly, which costs seconds per
# call. Compiling the step once keeps the loops that touch single games (UCI
# translation, interactive play, batch rollouts) cheap.
_step_one = jax.jit(env.step)
_step_many = jax.jit(jax.vmap(env.step))
_init_many = jax.jit(jax.vmap(env.init))

_DEFAULT_PLAYER_ORDER = jnp.array([0, 1], dtype=jnp.int32)


def player_id_for_colour(state, colour: int) -> int:
    """Return the player id that controls ``colour`` in this environment."""
    if colour not in (WHITE, BLACK):
        raise ValueError("colour must be 0 (white) or 1 (black)")
    order = jnp.asarray(getattr(state, "_player_order", _DEFAULT_PLAYER_ORDER), dtype=jnp.int32)
    return int(order[colour])


def board_of(state) -> chess.Board:
    """Return the absolute python-chess board for a PGX state."""
    return chess.Board(to_fen(state))


def legal_action_indices(state) -> list[int]:
    """Return the indices of every legal action in ``state``."""
    mask = jnp.asarray(state.legal_action_mask)
    return [int(index) for index in jnp.flatnonzero(mask).tolist()]


def legal_ucis(state) -> list[str]:
    """Return the legal moves of ``state`` in UCI notation."""
    return [move.uci() for move in board_of(state).legal_moves]


def _board_fen_after(state, action: int) -> str:
    return board_of(_step_one(state, action)).board_fen()


def action_to_uci(state, action: int) -> str | None:
    """Translate a PGX action into UCI, or ``None`` when it cannot be matched.

    PGX actions encode the move internally, so the move is recovered by playing
    every legal move of the position and comparing the resulting piece layout.
    Promotions are distinguished because the promoted piece changes that layout.
    """
    board = board_of(state)
    target = _board_fen_after(state, action)

    for move in board.legal_moves:
        probe = board.copy()
        probe.push(move)
        if probe.board_fen() == target:
            return move.uci()
    return None


def uci_to_action(state, uci: str) -> int | None:
    """Translate a UCI string into a PGX action, or ``None`` when it is illegal."""
    board = board_of(state)
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None

    if move not in board.legal_moves:
        return None

    probe = board.copy()
    probe.push(move)
    target = probe.board_fen()

    for action in legal_action_indices(state):
        if _board_fen_after(state, action) == target:
            return action
    return None


def reset_environments(key, num_envs: int):
    """Initialise ``num_envs`` independent games."""
    return _init_many(jax.random.split(key, num_envs))


def step_environment(state, action):
    """Advance a single game by one action."""
    return _step_one(state, action)


def step_environments(state, action):
    """Advance every environment by one action."""
    return _step_many(state, action)


@jax.jit
def reset_terminated(state, key):
    """Restart the finished games and leave the running ones untouched."""

    def select(fresh_leaf, current_leaf):
        mask = state.terminated
        for _ in range(fresh_leaf.ndim - 1):
            mask = mask[..., None]
        return jnp.where(mask, fresh_leaf, current_leaf)

    fresh = reset_environments(key, state.observation.shape[0])
    return jax.tree_util.tree_map(select, fresh, state)
