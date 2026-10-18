from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("pgx")
pytest.importorskip("chess")
pytest.importorskip("optax")

from chessrl.board import reset_environments  # noqa: E402
from chessrl.checkpoint import CheckpointError, load_params, save_params  # noqa: E402
from chessrl.config import TrainingConfig  # noqa: E402
from chessrl.model import NUM_ACTIONS, OBSERVATION_SHAPE, ChessBrain  # noqa: E402
from chessrl.training import build_optimizer, compute_loss, run_training, train_step_fn  # noqa: E402

NUM_ENVS = 2


@pytest.fixture(scope="module")
def params():
    observation = jnp.zeros((NUM_ENVS, *OBSERVATION_SHAPE), dtype=jnp.float32)
    mask = jnp.ones((NUM_ENVS, NUM_ACTIONS), dtype=jnp.bool_)
    return ChessBrain().init(jax.random.PRNGKey(0), observation, mask)["params"]


@pytest.fixture(scope="module")
def state():
    return reset_environments(jax.random.PRNGKey(42), NUM_ENVS)


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    return TrainingConfig(
        num_envs=NUM_ENVS,
        batches=1,
        save_every=1,
        log_every=1,
        checkpoint=tmp_path_factory.mktemp("checkpoints") / "brain.msgpack",
    )


class TestLoss:
    def test_the_loss_is_finite_and_returns_the_next_state(self, params, state):
        loss, next_state = compute_loss(
            params, params, state, jax.random.PRNGKey(1), jnp.float32(0.0)
        )

        assert jnp.isfinite(loss)
        assert next_state._step_count.shape == (NUM_ENVS,)
        assert jnp.all(next_state._step_count == state._step_count + 1)

    def test_full_exploration_still_produces_legal_moves(self, params, state):
        _, next_state = compute_loss(
            params, params, state, jax.random.PRNGKey(2), jnp.float32(1.0)
        )

        assert jnp.all(next_state._step_count == 1)
        assert jnp.any(next_state.legal_action_mask)

    def test_the_gradients_are_finite(self, params, state):
        grads = jax.grad(lambda p: compute_loss(p, p, state, jax.random.PRNGKey(3), 0.0)[0])(params)

        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))


class TestTrainStep:
    def test_one_step_returns_fresh_parameters_and_counters(self, params, state, config):
        optimizer = build_optimizer(config)
        train_step = train_step_fn(config, optimizer)
        opt_state = optimizer.init(params)

        updated, new_opt_state, next_state, loss, wins, draws, losses = train_step(
            params, params, opt_state, state, jax.random.PRNGKey(4), jnp.float32(0.5)
        )

        assert any(
            not jnp.allclose(a, b)
            for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(updated))
        )
        assert new_opt_state is not None
        assert jnp.isfinite(loss)
        assert int(wins + draws + losses) <= NUM_ENVS
        assert next_state._step_count.shape == (NUM_ENVS,)


class TestRunTraining:
    def test_a_short_run_writes_a_loading_checkpoint(self, config, params):
        messages = []
        path = run_training(config, log=messages.append)

        assert path == config.checkpoint
        assert path.is_file() and path.stat().st_size > 0
        assert load_params(params, path) is not None
        assert any("Training finished" in message for message in messages)

    def test_a_missing_resume_file_is_reported(self, tmp_path, params):
        config = TrainingConfig(
            num_envs=NUM_ENVS,
            batches=1,
            save_every=1,
            log_every=1,
            checkpoint=tmp_path / "brain.msgpack",
            resume_from=tmp_path / "absent.msgpack",
        )

        with pytest.raises(CheckpointError, match="not found"):
            run_training(config, log=lambda _message: None)

    def test_resuming_from_a_saved_checkpoint_works(self, config, params, tmp_path):
        save_params(params, tmp_path / "resume.msgpack")
        resumed = TrainingConfig(
            num_envs=NUM_ENVS,
            batches=1,
            save_every=1,
            log_every=1,
            checkpoint=config.checkpoint,
            resume_from=tmp_path / "resume.msgpack",
        )

        messages = []
        run_training(resumed, log=messages.append)

        assert any("Resumed the weights" in message for message in messages)
