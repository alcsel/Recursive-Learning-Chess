import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
import optax
import pgx
import os
from functools import partial

class ResidualBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = x + residual
        x = nn.relu(x)
        return x

class ChessBrain(nn.Module):
    num_blocks: int = 8
    features: int = 128

    @nn.compact
    def __call__(self, observation, legal_action_mask):
        x = observation.astype(jnp.float32)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        for _ in range(self.num_blocks):
            x = ResidualBlock(features=self.features)(x)
        x = x.reshape((x.shape[0], -1))
        logits = nn.Dense(features=4672)(x)
        logits = jnp.where(legal_action_mask, logits, -1e9)
        value = nn.Dense(features=1)(x)
        return logits, value

env = pgx.make("chess")

@jax.jit
def get_material_score(observation, current_player):
    weights = jnp.array([1., 3., 3., 5., 9., 0.], dtype=jnp.float32)
    is_black = (current_player == 1).astype(jnp.float32)
    white_ch = observation[:, :, :, 0:6]
    black_ch = observation[:, :, :, 6:12]
    my_pieces = jnp.where(is_black[:, None, None, None] > 0, black_ch, white_ch)
    opp_pieces = jnp.where(is_black[:, None, None, None] > 0, white_ch, black_ch)
    return jnp.sum(my_pieces * weights, axis=(1, 2, 3)) - jnp.sum(opp_pieces * weights, axis=(1, 2, 3))

@partial(jax.jit, static_argnames=['num_envs'])
def reset_environments(key, num_envs):
    keys = jax.random.split(key, num_envs)
    return jax.vmap(env.init)(keys)

@jax.jit
def step_environments(state, action):
    return jax.vmap(env.step)(state, action)

@jax.jit
def reset_terminated(state, key):
    # Biten ortamlari sifirla, devam edenlere dokunma
    num_envs = state.observation.shape[0]
    keys = jax.random.split(key, num_envs)
    fresh = jax.vmap(env.init)(keys)
    def select(new_val, old_val):
        mask = state.terminated
        # broadcast mask to match array shape
        for _ in range(new_val.ndim - 1):
            mask = mask[..., None]
        return jnp.where(mask, new_val, old_val)
    return jax.tree_util.tree_map(select, fresh, state)

LEARNING_RATE = 0.0001
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(LEARNING_RATE)
)

@jax.jit
def train_step(params, frozen_params, opt_state, state, key, epsilon):
    def loss_fn(p):
        is_white = (state.current_player == 0)

        white_logits, white_value = ChessBrain().apply({'params': p}, state.observation, state.legal_action_mask)
        white_value = white_value.squeeze(-1)

        frozen_logits, _ = ChessBrain().apply({'params': frozen_params}, state.observation, state.legal_action_mask)

        white_action = jax.random.categorical(key, logits=white_logits, axis=-1)

        frozen_black_action = jax.random.categorical(key, logits=frozen_logits, axis=-1)
        random_logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        rng_action = jax.random.categorical(key, logits=random_logits, axis=-1)
        num_envs = state.observation.shape[0]
        use_random = jax.random.uniform(key, shape=(num_envs,)) < epsilon
        black_action = jnp.where(use_random, rng_action, frozen_black_action)

        action = jnp.where(is_white, white_action, black_action)

        white_log_probs = jax.nn.log_softmax(white_logits)
        action_log_prob = jnp.take_along_axis(white_log_probs, action[..., None], axis=-1).squeeze(-1)

        next_state = step_environments(state, action)

        _, next_value_raw = ChessBrain().apply({'params': p}, next_state.observation, next_state.legal_action_mask)
        next_value_raw = next_value_raw.squeeze(-1)
        next_value = -next_value_raw

        raw_game_reward = jnp.take_along_axis(
            next_state.rewards, state.current_player[:, None], axis=-1
        ).squeeze(-1)

        game_reward = raw_game_reward

        current_mat = get_material_score(state.observation, state.current_player)
        next_mat = get_material_score(next_state.observation, next_state.current_player)
        material_gain = jnp.clip((current_mat - next_mat) / 39.0, -1.0, 1.0) * 0.3

        enemy_mobility = jnp.sum(next_state.legal_action_mask, axis=-1, dtype=jnp.float32)
        claustrophobia_penalty = -(enemy_mobility / 4672.0) * 0.2

        is_stalemate_trap = next_state.terminated & (raw_game_reward == 0.0) & (current_mat > 5.0)
        generic_draw = next_state.terminated & (raw_game_reward == 0.0) & (~is_stalemate_trap)
        draw_penalty = jnp.where(is_stalemate_trap, -1.0, 0.0)
        draw_penalty = draw_penalty + jnp.where(generic_draw, -0.5, 0.0)

        step_penalty = -0.001

        reward = game_reward + material_gain + claustrophobia_penalty + draw_penalty + step_penalty

        gamma = 0.99
        is_terminated = next_state.terminated.astype(jnp.float32)
        td_target = reward + gamma * next_value * (1.0 - is_terminated)
        td_target = jnp.clip(td_target, -5.0, 5.0)
        advantage = td_target - white_value

        is_white_f = is_white.astype(jnp.float32)
        actor_loss = -jnp.mean(action_log_prob * jax.lax.stop_gradient(advantage) * is_white_f)
        critic_loss = 0.5 * jnp.mean(jnp.square(advantage) * is_white_f)

        probs = jax.nn.softmax(white_logits)
        entropy = -jnp.sum(jnp.where(probs > 0, probs * white_log_probs, 0.0), axis=-1)
        entropy_loss = -0.01 * jnp.mean(entropy * is_white_f)

        total_loss = actor_loss + critic_loss + entropy_loss
        return total_loss, (next_state, raw_game_reward)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (next_state, raw_game_reward)), grads = grad_fn(params)

    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    wins = jnp.sum(raw_game_reward == 1.0)
    losses = jnp.sum(raw_game_reward == -1.0)
    draws = jnp.sum(next_state.terminated & (raw_game_reward == 0.0))

    return new_params, new_opt_state, next_state, loss, wins, draws, losses

def get_epsilon(batch_idx):
    if batch_idx < 1000:
        return 0.5
    elif batch_idx < 3000:
        return 0.2
    else:
        return 0.05

FROZEN_GUNCELLEME = 500

if __name__ == "__main__":
    print("[SISTEM] 8 Residual Block + Auto-Reset modeli baslatiliyor.")

    NUM_ENVS = 2048
    BATCHES = 100000
    YUKLE_YOLU = "/kaggle/input/datasets/selimalc/satran-148k/satranc_beyni_jax 148k.msgpack"
    KAYIT_YOLU = "/kaggle/working/satranc_beyni_jax.msgpack"
    os.makedirs(os.path.dirname(KAYIT_YOLU), exist_ok=True)

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    model = ChessBrain()
    dummy_obs = jnp.zeros((NUM_ENVS, 8, 8, 119), dtype=jnp.float32)
    dummy_mask = jnp.ones((NUM_ENVS, 4672), dtype=jnp.bool_)

    params = model.init(init_rng, dummy_obs, dummy_mask)['params']
    opt_state = optimizer.init(params)

    if os.path.exists(YUKLE_YOLU):
        try:
            with open(YUKLE_YOLU, "rb") as f:
                params = serialization.from_bytes(params, f.read())
            print(f"[BILGI] Agirliklar yuklendi: {YUKLE_YOLU}")
        except Exception as e:
            print(f"[UYARI] Kayit okunamadi. Hata: {e}")

    frozen_params = serialization.from_bytes(params, serialization.to_bytes(params))

    rng, reset_rng = jax.random.split(rng)
    state = reset_environments(reset_rng, num_envs=NUM_ENVS)

    print(f"[BILGI] {NUM_ENVS} ortam derlendi. Antrenman basliyor...")

    total_wins = 0
    total_draws = 0
    total_losses = 0

    try:
        for batch_idx in range(1, BATCHES + 1):
            rng, step_rng, reset_rng = jax.random.split(rng, 3)

            state = reset_terminated(state, reset_rng)

            epsilon = jnp.float32(get_epsilon(batch_idx))
            params, opt_state, state, loss, wins, draws, losses = train_step(
                params, frozen_params, opt_state, state, step_rng, epsilon
            )

            total_wins += int(wins.item())
            total_draws += int(draws.item())
            total_losses += int(losses.item())

            if batch_idx % FROZEN_GUNCELLEME == 0:
                frozen_params = serialization.from_bytes(params, serialization.to_bytes(params))
                print(f"[BILGI] Frozen opponent guncellendi. Epsilon: {get_epsilon(batch_idx):.2f} (Batch {batch_idx})")

            if batch_idx % 10 == 0:
                print(f"Batch: {batch_idx:06d} | Kayip: {loss.item():.4f} | K-B-M: {wins.item():03d}-{draws.item():03d}-{losses.item():03d} | Toplam: {total_wins}-{total_draws}-{total_losses} | Eps: {get_epsilon(batch_idx):.2f}")

            if batch_idx % 25 == 0:
                with open(KAYIT_YOLU, "wb") as f:
                    f.write(serialization.to_bytes(params))

    except KeyboardInterrupt:
        with open(KAYIT_YOLU, "wb") as f:
            f.write(serialization.to_bytes(params))
        print(f"[BILGI] Ctrl+C algilandi, model kaydedildi. Toplam K-B-M: {total_wins}-{total_draws}-{total_losses}")
        exit()

    with open(KAYIT_YOLU, "wb") as f:
        f.write(serialization.to_bytes(params))
    print("[BILGI] Egitim tamamlandi.")
