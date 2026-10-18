from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nn = pytest.importorskip("flax.linen")

from chessrl.model import NUM_ACTIONS, OBSERVATION_SHAPE, ChessBrain, ResidualBlock  # noqa: E402

MASKED_LOGIT = -1e9


@pytest.fixture(scope="module")
def variables():
    observation = jnp.zeros((1, *OBSERVATION_SHAPE), dtype=jnp.float32)
    mask = jnp.zeros((1, NUM_ACTIONS), dtype=jnp.bool_).at[0, 3].set(True)
    return ChessBrain().init(jax.random.PRNGKey(0), observation, mask), observation, mask


class TestArchitecture:
    def test_the_head_shapes_match_the_environment(self, variables):
        params, observation, mask = variables
        logits, value = ChessBrain().apply({"params": params["params"]}, observation, mask)

        assert logits.shape == (1, NUM_ACTIONS)
        assert value.shape == (1, 1)

    def test_illegal_actions_are_masked_out(self, variables):
        params, observation, mask = variables
        logits, _ = ChessBrain().apply({"params": params["params"]}, observation, mask)

        assert float(logits[0, 3]) > MASKED_LOGIT
        assert float(logits[0, 0]) == MASKED_LOGIT

    def test_the_default_depth_matches_the_documented_model(self):
        brain = ChessBrain()
        assert brain.num_blocks == 8
        assert brain.features == 128

    def test_a_residual_block_keeps_the_input_shape(self):
        block = ResidualBlock(features=16)
        x = jnp.zeros((1, 8, 8, 16), dtype=jnp.float32)
        out = block.apply(block.init(jax.random.PRNGKey(0), x), x)
        assert out.shape == x.shape
