"""Play a game against a trained checkpoint."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from .board import (
    BLACK,
    WHITE,
    action_to_uci,
    board_of,
    env,
    legal_ucis,
    player_id_for_colour,
    step_environment,
    uci_to_action,
)
from .checkpoint import load_params
from .config import PlayConfig, colour_name
from .model import NUM_ACTIONS, OBSERVATION_SHAPE, ChessBrain

__all__ = ["play"]


def _load_model(checkpoint):
    """Build the network template and load the checkpoint over it.

    The apply function is compiled once so that querying the network during a
    game does not pay the eager dispatch cost per move.
    """
    model = ChessBrain()
    dummy_obs = jnp.zeros((1, *OBSERVATION_SHAPE), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, NUM_ACTIONS), dtype=jnp.bool_)
    template = model.init(jax.random.PRNGKey(0), dummy_obs, dummy_mask)["params"]
    return jax.jit(model.apply), load_params(template, checkpoint)


def _model_action(apply, params, state) -> tuple[int, float]:
    logits, value = apply(
        {"params": params},
        state.observation[None],
        state.legal_action_mask[None],
    )
    return int(jnp.argmax(logits[0])), float(value[0, 0])


def _ask_human(state) -> int | None:
    """Read one move from stdin, or ``None`` when the player stops the game."""
    print(f"Legal moves: {' '.join(legal_ucis(state))}")

    while True:
        try:
            answer = input("Your move (UCI, 'help' for the board, 'quit' to stop): ").strip()
        except EOFError:
            print()
            return None

        if answer.lower() in {"quit", "exit"}:
            return None
        if answer.lower() in {"help", "moves", "?", "board"}:
            print(board_of(state))
            continue

        action = uci_to_action(state, answer)
        if action is None:
            print(f"Illegal move: {answer}")
            continue
        return action


def _report_result(state, log=print) -> None:
    white_reward = float(state.rewards[player_id_for_colour(state, WHITE)])
    black_reward = float(state.rewards[player_id_for_colour(state, BLACK)])

    if white_reward == 0.0 and black_reward == 0.0:
        log("Game over: draw (stalemate, repetition, insufficient material or the 512 step limit).")
    elif white_reward > 0.0:
        log("Game over: White wins.")
    else:
        log("Game over: Black wins.")


def play(config: PlayConfig | None = None, *, log=print) -> None:
    """Run one interactive game between the human and the checkpoint."""
    config = config or PlayConfig()
    apply, params = _load_model(config.checkpoint)

    human_colour = config.human_colour
    state = env.init(jax.random.PRNGKey(config.seed))
    model_player = player_id_for_colour(state, config.model_colour)
    human_player = player_id_for_colour(state, human_colour)

    log(f"Loaded the weights from {config.checkpoint}")
    log(f"You play {colour_name(human_colour)}; the model plays {colour_name(config.model_colour)}.")

    while not bool(state.terminated):
        if config.show_board:
            print(board_of(state))

        if int(state.current_player) == model_player:
            action, value = _model_action(apply, params, state)
            uci = action_to_uci(state, action) or f"unrecognised action {action}"
            print(f"Model ({colour_name(config.model_colour)}): {uci} [value {value:+.3f}]")
        else:
            assert int(state.current_player) == human_player
            action = _ask_human(state)
            if action is None:
                log("Game stopped.")
                return

        state = step_environment(state, action)

    if config.show_board:
        print(board_of(state))
    _report_result(state, log=log)


def main() -> int:
    """Entry point for ``python -m chessrl.play``."""
    from .cli import main as cli_main

    return cli_main(["play", *sys.argv[1:]])
