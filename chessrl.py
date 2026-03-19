import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
import pgx
from pgx.experimental.chess import to_fen
import chess

class ResidualBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=self.features, kernel_size=(3,3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3,3), padding='SAME')(x)
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
        x = nn.Conv(features=self.features, kernel_size=(3,3), padding='SAME')(x)
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

def model_action_to_uci(action, state):
    fen_before = to_fen(state)
    board_before = chess.Board(fen_before)
    next_s = env.step(state, action)
    fen_after = to_fen(next_s)
    pieces_after = chess.Board(fen_after).board_fen()
    for move in board_before.legal_moves:
        b = board_before.copy()
        b.push(move)
        if b.board_fen() == pieces_after:
            return move.uci()
    return f"?({action})"

def user_uci_to_action(uci, state):
    fen = to_fen(state)
    board = chess.Board(fen)
    legal_ucis = [m.uci() for m in board.legal_moves]
    if uci not in legal_ucis:
        return None, legal_ucis
    move = chess.Move.from_uci(uci)
    b_after = board.copy()
    b_after.push(move)
    pieces_after = b_after.board_fen()
    indices = [i for i in range(4672) if state.legal_action_mask[i]]
    for idx in indices:
        next_s = env.step(state, idx)
        if chess.Board(to_fen(next_s)).board_fen() == pieces_after:
            return idx, legal_ucis
    return None, legal_ucis

model_nn = ChessBrain()
dummy_obs = jnp.zeros((1, 8, 8, 119))
dummy_mask = jnp.ones((1, 4672), dtype=jnp.bool_)
params = model_nn.init(jax.random.PRNGKey(0), dummy_obs, dummy_mask)['params']

with open(r"D:\Code\vscode\python\chessai\chessrl\satranc_beyni_jax 112k.msgpack", "rb") as f:
    params = serialization.from_bytes(params, f.read())

rng = jax.random.PRNGKey(42)
state = env.init(rng)

print("Sen Siyah, Model Beyaz.")

while not state.terminated:
    current_player = int(state.current_player)

    if current_player == 1:
        obs = state.observation[None]
        mask = state.legal_action_mask[None]
        logits, value = model_nn.apply({'params': params}, obs, mask)
        action = int(jnp.argmax(logits[0]))
        uci = model_action_to_uci(action, state)
        rng, step_rng = jax.random.split(rng)
        state = env.step(state, action)
        print(f"\nModel (Beyaz): {uci} (value: {value[0,0]:.3f})")
    else:
        fen = to_fen(state)
        board = chess.Board(fen)
        legal_ucis = [m.uci() for m in board.legal_moves]
        print(f"Yasal hamleler: {' '.join(legal_ucis)}")
        uci = input("Hamlen (UCI): ").strip()
        action, _ = user_uci_to_action(uci, state)
        if action is None:
            print("Gecersiz hamle!")
            continue
        rng, step_rng = jax.random.split(rng)
        state = env.step(state, action)

print(f"\nOyun bitti!")
print(f"Kazanan: {'Beyaz' if state.rewards[1] > 0 else 'Siyah' if state.rewards[0] > 0 else 'Berabere'}")