"""Self-play training.

The trainee (the parameters that receive gradients) always occupies the same
player slot and the frozen opponent the other one. PGX decides at random which
colour a slot controls, so across a batch of games the trainee plays both
colours, which is what we want from a single network.
"""

from __future__ import annotations

import copy
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from .board import reset_environments, reset_terminated, step_environments
from .checkpoint import load_params, save_params
from .config import TrainingConfig, epsilon_for
from .model import NUM_ACTIONS, OBSERVATION_SHAPE, ChessBrain
from .rewards import material_after_own_move, material_balance, material_gain, shaped_reward, td_target

__all__ = ["OPPONENT_PLAYER", "TRAINEE_PLAYER", "build_optimizer", "compute_loss", "run_training", "train_step_fn"]

TRAINEE_PLAYER = 0
OPPONENT_PLAYER = 1


def build_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """Global norm clipping followed by Adam."""
    return optax.chain(
        optax.clip_by_global_norm(config.clip_by_global_norm),
        optax.adam(config.learning_rate),
    )


def compute_loss(
    params,
    frozen_params,
    state,
    key,
    epsilon,
    *,
    gamma: float = 0.99,
    entropy_coefficient: float = 0.01,
):
    """One advantage-actor-critic step, averaged over the trainee's moves.

    Returns the scalar loss and the next state, so the caller can keep rolling
    the environments forward and can report the finished games. ``gamma`` and
    ``entropy_coefficient`` are plain Python floats so the whole function stays
    traceable by ``jax.jit``.
    """
    trainee_moves = state.current_player == TRAINEE_PLAYER
    moving = trainee_moves.astype(jnp.float32)

    trainee_logits, trainee_value = ChessBrain().apply(
        {"params": params}, state.observation, state.legal_action_mask
    )
    trainee_value = trainee_value.squeeze(-1)

    opponent_logits, _ = ChessBrain().apply(
        {"params": frozen_params}, state.observation, state.legal_action_mask
    )

    key, trainee_key, opponent_key, explore_key = jax.random.split(key, 4)

    trainee_action = jax.random.categorical(trainee_key, logits=trainee_logits, axis=-1)

    exploration_logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
    explore_action = jax.random.categorical(explore_key, logits=exploration_logits, axis=-1)
    exploit_action = jax.random.categorical(opponent_key, logits=opponent_logits, axis=-1)
    exploring = jax.random.uniform(explore_key, shape=(state.observation.shape[0],)) < epsilon
    opponent_action = jnp.where(exploring, explore_action, exploit_action)

    action = jnp.where(trainee_moves, trainee_action, opponent_action)

    log_probs = jax.nn.log_softmax(trainee_logits)
    action_log_prob = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)

    next_state = step_environments(state, action)

    _, next_value_raw = ChessBrain().apply(
        {"params": params}, next_state.observation, next_state.legal_action_mask
    )
    next_value = -next_value_raw.squeeze(-1)

    game_reward = jnp.take_along_axis(
        next_state.rewards, state.current_player[:, None], axis=-1
    ).squeeze(-1)

    current_balance = material_balance(state.observation)
    next_balance = material_after_own_move(next_state.observation)
    gain = material_gain(current_balance, next_balance)

    enemy_mobility = jnp.sum(next_state.legal_action_mask, axis=-1, dtype=jnp.float32)
    reward = shaped_reward(
        game_reward=game_reward,
        gain=gain,
        enemy_mobility=enemy_mobility,
        terminated=next_state.terminated,
        current_balance=current_balance,
        num_actions=NUM_ACTIONS,
    )

    advantage = td_target(reward, next_value, next_state.terminated, gamma=gamma) - trainee_value

    actor_loss = -jnp.mean(action_log_prob * jax.lax.stop_gradient(advantage) * moving)
    critic_loss = 0.5 * jnp.mean(jnp.square(advantage) * moving)

    probs = jax.nn.softmax(trainee_logits)
    entropy = -jnp.sum(jnp.where(probs > 0, probs * log_probs, 0.0), axis=-1)
    entropy_loss = -entropy_coefficient * jnp.mean(entropy * moving)

    return actor_loss + critic_loss + entropy_loss, next_state


def train_step_fn(config: TrainingConfig, optimizer: optax.GradientTransformation):
    """Build the jitted update step, closing over the config and the optimizer."""

    gamma = config.gamma
    entropy_coefficient = config.entropy_coefficient

    @jax.jit
    def train_step(params, frozen_params, opt_state, state, key, epsilon):
        (loss, next_state), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            params,
            frozen_params,
            state,
            key,
            epsilon,
            gamma=gamma,
            entropy_coefficient=entropy_coefficient,
        )

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        game_reward = jnp.take_along_axis(
            next_state.rewards, state.current_player[:, None], axis=-1
        ).squeeze(-1)
        wins = jnp.sum(game_reward == 1.0)
        losses = jnp.sum(game_reward == -1.0)
        draws = jnp.sum(next_state.terminated & (game_reward == 0.0))

        return new_params, new_opt_state, next_state, loss, wins, draws, losses

    return train_step


def _initial_params(config: TrainingConfig, key):
    model = ChessBrain()
    dummy_obs = jnp.zeros((config.num_envs, *OBSERVATION_SHAPE), dtype=jnp.float32)
    dummy_mask = jnp.ones((config.num_envs, NUM_ACTIONS), dtype=jnp.bool_)
    params = model.init(key, dummy_obs, dummy_mask)["params"]

    if config.resume_from is not None:
        params = load_params(params, config.resume_from)
    return params


def run_training(config: TrainingConfig | None = None, *, log=print) -> Path:
    """Train the network and return the path of the final checkpoint."""
    config = config or TrainingConfig()
    optimizer = build_optimizer(config)

    key = jax.random.PRNGKey(config.seed)
    key, init_key, reset_key = jax.random.split(key, 3)
    params = _initial_params(config, init_key)
    opt_state = optimizer.init(params)

    if config.resume_from is not None:
        log(f"Resumed the weights from {config.resume_from}")

    frozen_params = copy.deepcopy(params)
    state = reset_environments(reset_key, config.num_envs)
    train_step = train_step_fn(config, optimizer)

    log(f"Training {config.batches} batches on {config.num_envs} parallel games")

    totals = {"wins": 0, "draws": 0, "losses": 0}
    try:
        for batch in range(1, config.batches + 1):
            key, step_key, reset_key = jax.random.split(key, 3)
            state = reset_terminated(state, reset_key)
            epsilon = epsilon_for(batch)

            params, opt_state, state, loss, wins, draws, losses = train_step(
                params, frozen_params, opt_state, state, step_key, jnp.float32(epsilon)
            )

            totals["wins"] += int(wins)
            totals["draws"] += int(draws)
            totals["losses"] += int(losses)

            if config.should_update_opponent(batch):
                frozen_params = copy.deepcopy(params)
                log(f"Frozen opponent updated at batch {batch} (epsilon {epsilon:.2f})")

            if config.should_log(batch):
                log(
                    f"Batch {batch:06d} | Loss {float(loss):.4f} | "
                    f"W-D-L {int(wins):03d}-{int(draws):03d}-{int(losses):03d} | "
                    f"Total {totals['wins']}-{totals['draws']}-{totals['losses']} | "
                    f"Eps {epsilon:.2f}"
                )

            if config.should_save(batch):
                save_params(params, config.checkpoint)
    except KeyboardInterrupt:
        path = save_params(params, config.checkpoint)
        log(f"Interrupted; the weights were saved to {path}")
        raise SystemExit(130) from None

    path = save_params(params, config.checkpoint)
    log(f"Training finished; the weights were saved to {path}")
    return path
