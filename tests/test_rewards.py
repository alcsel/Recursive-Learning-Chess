from __future__ import annotations

import pytest

jnp = pytest.importorskip("jax.numpy")

from chessrl.rewards import (  # noqa: E402
    DRAW_PENALTY,
    MATERIAL_WEIGHT,
    PIECE_VALUES,
    STALEMATE_PENALTY,
    STEP_PENALTY,
    material_after_own_move,
    material_balance,
    material_gain,
    shaped_reward,
    td_target,
)

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
_OPPONENT_PLANE_OFFSET = 6

FULL_ARMY = (
    [(PAWN, square) for square in range(8)]
    + [(KNIGHT, 8), (KNIGHT, 9)]
    + [(BISHOP, 10), (BISHOP, 11)]
    + [(ROOK, 12), (ROOK, 13)]
    + [(QUEEN, 14)]
)


def _observation(own=(), opponents=()):
    """Build a one environment batch where each entry is ``(plane, square)``."""
    observation = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    for plane, square in own:
        observation = observation.at[0, square // 8, square % 8, plane].set(1.0)
    for plane, square in opponents:
        observation = observation.at[0, square // 8, square % 8, plane + _OPPONENT_PLANE_OFFSET].set(1.0)
    return observation


class TestMaterialBalance:
    def test_reads_the_movers_own_planes_first(self):
        balance = material_balance(_observation(own=[(QUEEN, 0)], opponents=[(ROOK, 1)]))
        assert float(balance[0]) == pytest.approx(9.0 - 5.0)

    def test_an_equal_army_balances_to_zero(self):
        army = [(PAWN, 0), (KNIGHT, 1), (BISHOP, 2), (ROOK, 3), (QUEEN, 4), (KING, 5)]
        assert float(material_balance(_observation(own=army, opponents=army))[0]) == 0.0

    def test_the_king_is_worth_nothing(self):
        assert float(material_balance(_observation(own=[(KING, 0)]))[0]) == 0.0

    def test_the_piece_values_are_the_usual_ones(self):
        assert PIECE_VALUES.tolist() == [1.0, 3.0, 3.0, 5.0, 9.0, 0.0]

    def test_the_next_observation_belongs_to_the_opponent(self):
        after = _observation(own=[(ROOK, 0)])
        assert float(material_after_own_move(after)[0]) == -5.0


class TestMaterialGain:
    def test_capturing_a_queen_is_rewarded(self):
        before = _observation(own=[(PAWN, 8)], opponents=[(QUEEN, 63)])
        after = _observation(own=[], opponents=[(PAWN, 8)])

        gain = material_gain(material_balance(before), material_after_own_move(after))

        assert float(gain[0]) == pytest.approx(MATERIAL_WEIGHT * 9.0 / 39.0)

    def test_a_move_without_a_capture_earns_nothing(self):
        position = _observation(own=[(PAWN, 8)])
        assert float(material_gain(material_balance(position), material_balance(position))[0]) == 0.0

    def test_the_bonus_is_capped_at_one_army(self):
        empty = _observation()
        full_army = _observation(own=FULL_ARMY)
        assert float(material_balance(full_army)[0]) == pytest.approx(39.0)
        assert float(material_gain(material_balance(empty), material_balance(full_army))[0]) == pytest.approx(
            MATERIAL_WEIGHT
        )

    def test_losing_material_is_punished(self):
        start = _observation(own=[(QUEEN, 0)])
        end = _observation()
        assert float(material_gain(material_balance(start), material_balance(end))[0]) < 0.0


class TestShapedReward:
    def _reward(self, **overrides):
        arguments = {
            "game_reward": jnp.float32(0.0),
            "gain": jnp.float32(0.0),
            "enemy_mobility": jnp.float32(0.0),
            "terminated": jnp.bool_(False),
            "current_balance": jnp.float32(0.0),
            "num_actions": 4672,
        }
        arguments.update(overrides)
        return float(shaped_reward(**arguments))

    def test_a_quiet_move_only_costs_the_step_penalty(self):
        assert self._reward() == pytest.approx(STEP_PENALTY)

    def test_winning_keeps_the_terminal_reward(self):
        assert self._reward(game_reward=jnp.float32(1.0), terminated=jnp.bool_(True)) == pytest.approx(
            1.0 + STEP_PENALTY
        )

    def test_a_generic_draw_is_penalised(self):
        assert self._reward(terminated=jnp.bool_(True)) == pytest.approx(DRAW_PENALTY + STEP_PENALTY)

    def test_throwing_away_a_winning_position_hurts_more(self):
        reward = self._reward(
            terminated=jnp.bool_(True),
            current_balance=jnp.float32(9.0),
        )
        assert reward == pytest.approx(STALEMATE_PENALTY + STEP_PENALTY)
        assert reward < DRAW_PENALTY

    def test_a_draw_without_a_material_lead_is_not_a_stalemate_trap(self):
        reward = self._reward(terminated=jnp.bool_(True), current_balance=jnp.float32(4.0))
        assert reward == pytest.approx(DRAW_PENALTY + STEP_PENALTY)

    def test_the_opponents_mobility_is_penalised(self):
        assert self._reward(enemy_mobility=jnp.float32(4672.0)) == pytest.approx(-0.2 + STEP_PENALTY)


class TestTdTarget:
    def test_a_terminal_state_ignores_the_bootstrap(self):
        target = td_target(jnp.float32(1.0), jnp.float32(-3.0), jnp.bool_(True))
        assert float(target) == pytest.approx(1.0)

    def test_a_running_state_bootstraps_with_gamma(self):
        target = td_target(jnp.float32(0.0), jnp.float32(1.0), jnp.bool_(False), gamma=0.99)
        assert float(target) == pytest.approx(0.99)

    def test_the_target_is_clipped(self):
        assert float(td_target(jnp.float32(4.0), jnp.float32(9.0), jnp.bool_(False))) == pytest.approx(5.0)
        assert float(td_target(jnp.float32(-4.0), jnp.float32(-9.0), jnp.bool_(False))) == pytest.approx(-5.0)
