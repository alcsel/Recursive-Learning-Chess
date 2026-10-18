from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("pgx")
chess = pytest.importorskip("chess")

from chessrl.board import (  # noqa: E402
    BLACK,
    WHITE,
    action_to_uci,
    board_of,
    env,
    legal_action_indices,
    legal_ucis,
    player_id_for_colour,
    reset_environments,
    reset_terminated,
    uci_to_action,
)

STARTING_PIECES = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


@pytest.fixture(scope="module")
def state():
    return env.init(jax.random.PRNGKey(0))


class TestPlayerOrder:
    def test_the_two_colours_map_to_the_two_player_slots(self, state):
        slots = {player_id_for_colour(state, WHITE), player_id_for_colour(state, BLACK)}
        assert slots == {0, 1}

    def test_white_is_whichever_slot_the_environment_assigned(self, state):
        assert player_id_for_colour(state, WHITE) == int(state._player_order[0])

    def test_rejects_an_unknown_colour(self, state):
        with pytest.raises(ValueError, match="colour"):
            player_id_for_colour(state, 2)


class TestBoardConversion:
    def test_the_initial_position_is_the_standard_one(self, state):
        board = board_of(state)
        assert board.board_fen() == STARTING_PIECES
        assert board.turn == chess.WHITE
        assert board.fen().split()[1] == "w"

    def test_white_moves_first_regardless_of_the_player_order(self, state):
        white_mover = board_of(state).turn == chess.WHITE
        assert white_mover == (int(state.current_player) == player_id_for_colour(state, WHITE))

    def test_the_starting_position_has_twenty_moves(self, state):
        assert len(legal_ucis(state)) == 20
        assert len(legal_action_indices(state)) == 20


class TestUciTranslation:
    def test_a_named_move_round_trips(self, state):
        action = uci_to_action(state, "e2e4")

        assert action is not None
        assert action in legal_action_indices(state)
        assert action_to_uci(state, action) == "e2e4"

    def test_every_legal_move_round_trips(self, state):
        for action in legal_action_indices(state):
            uci = action_to_uci(state, action)
            assert uci is not None
            assert uci_to_action(state, uci) == action

    def test_an_illegal_move_is_rejected(self, state):
        assert uci_to_action(state, "e2e5") is None

    def test_malformed_input_is_rejected(self, state):
        assert uci_to_action(state, "not-a-move") is None
        assert uci_to_action(state, "") is None

    @pytest.mark.parametrize("promotion", ["e7e8q", "e7e8r", "e7e8b", "e7e8n"])
    def test_promotions_are_distinguished(self, promotion):
        from pgx.experimental.chess import from_fen

        promotion_state = from_fen("7k/4P3/8/8/8/8/8/4K3 w - - 0 1")
        action = uci_to_action(promotion_state, promotion)

        assert action is not None
        assert action_to_uci(promotion_state, action) == promotion


class TestEnvironmentBatching:
    def test_reset_builds_one_game_per_environment(self):
        states = reset_environments(jax.random.PRNGKey(1), 3)

        assert states.observation.shape == (3, 8, 8, 119)
        assert states.legal_action_mask.shape == (3, 4672)

    def test_reset_terminated_only_restarts_finished_games(self):
        states = reset_environments(jax.random.PRNGKey(2), 3)
        actions = jnp.argmax(states.legal_action_mask, axis=-1)
        advanced = jax.vmap(env.step)(states, actions)

        marked = advanced.replace(terminated=advanced.terminated.at[1].set(True))
        restarted = reset_terminated(marked, jax.random.PRNGKey(3))

        assert int(restarted._step_count[1]) == 0
        assert int(restarted._step_count[0]) == 1
        assert int(restarted._step_count[2]) == 1
