# Chess RL Agent — JAX/Flax Self-Play

A chess reinforcement learning agent trained via self-play using JAX and Flax on the PGX chess environment.

## Architecture

- **Model**: `ChessBrain` — 8 Residual Blocks, 128 features
- **Normalization**: LayerNorm (not BatchNorm)
- **Heads**: Policy (4672 actions) + Value (scalar)
- **Optimizer**: Adam (lr=0.0001) + gradient clipping (norm=1.0)

### Residual Block

```
Conv → LayerNorm → ReLU → Conv → LayerNorm → Skip → ReLU
```

### Input / Output

| | Shape |
|---|---|
| Observation | (N, 8, 8, 119) |
| Legal action mask | (N, 4672) |
| Policy logits | (N, 4672) |
| Value | (N, 1) |

## Training

### Setup

- **Environment**: PGX Chess
- **Parallel envs**: 2048
- **Hardware**: Kaggle T4 GPU
- **Total batches**: 70k+

### Self-Play

- **White**: current `params` (gradients flow)
- **Black**: `frozen_params` (no gradients, updated every 500 batches)
- **Opponent update**: deep copy via `serialization.to_bytes` / `from_bytes`

### Exploration

| Batch range | Epsilon |
|---|---|
| 0 – 1000 | 0.50 |
| 1000 – 3000 | 0.20 |
| 3000+ | 0.05 |

### Reward Shaping

| Component | Formula |
|---|---|
| Game result | {-1, 0, 1} |
| Material gain | `clip((cur_mat - next_mat) / 39.0, -1, 1) * 0.3` |
| Mobility penalty | `-(enemy_moves / 4672.0) * 0.2` |
| Stalemate trap | -1.0 |
| Generic draw | -0.5 |
| Step penalty | -0.001 |

TD target clipped to [-5, 5].

### Loss Function

```
actor_loss  = -mean(log_prob * stop_grad(advantage) * is_white)
critic_loss = 0.5 * mean(advantage² * is_white)
entropy_loss = -0.01 * mean(entropy * is_white)
total_loss  = actor_loss + critic_loss + entropy_loss
```

## Results

Win/draw ratio over training:

| Batch | Win/Draw ratio |
|---|---|
| 17k | ~1.00 |
| 24k | ~1.05 |
| 26k | ~1.08 |
| 72k+ | ~13.0 |

The model learns to prefer decisive results over draws as training progresses.

## Installation

```bash
pip install pgx flax jax chess
```

For GPU support:
```bash
pip install jax[cuda12]
```

## Play Against the Model

Download the checkpoint from the release and place it next to `chessrl.py`.

```bash
python chessrl.py
```

- You play as **Black**
- Model plays as **White**
- Input moves in UCI format: `e7e5`, `g8f6`, etc.

## File Structure

```
satranc_jax_fixed.py   # Training script
chessrl.py             # Play against the model
satranc_beyni_jax.msgpack  # Model checkpoint
```

## Notes

- No MCTS — pure policy inference
- Strong ELO requires 100k–200k+ batches with MCTS
- Windows users: `uvloop` not supported, install with `--no-deps`

## License

MIT
