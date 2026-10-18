# Recursive Learning Chess

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![JAX](https://img.shields.io/badge/JAX-Flax-9cf)
![License](https://img.shields.io/badge/license-MIT-green)

A chess reinforcement learning agent trained purely by self-play with JAX,
Flax and the [PGX](https://github.com/sotetsuk/pgx) chess environment. Two
instances of the same residual network play each other, one of them frozen as a
periodically refreshed opponent, and the policy learns from the game result plus
a small amount of reward shaping.

No search is involved: the agent plays by policy inference alone.

## Contents

- [Architecture](#architecture)
- [Training](#training)
- [Quick start](#quick-start)
- [Playing against the model](#playing-against-the-model)
- [Project layout](#project-layout)
- [Correctness notes](#correctness-notes)
- [Tests](#tests)
- [License](#license)

## Architecture

- **Model**: `ChessBrain` — 8 residual blocks, 128 features
- **Normalisation**: LayerNorm (not BatchNorm), which behaves the same in training and inference
- **Heads**: policy (4672 actions) and value (scalar)
- **Optimizer**: Adam (lr = 1e-4) with global norm clipping at 1.0

### Residual block

```
Conv → LayerNorm → ReLU → Conv → LayerNorm → (+ skip) → ReLU
```

### Input and output

| | Shape |
|---|---|
| Observation | (N, 8, 8, 119) |
| Legal action mask | (N, 4672) |
| Policy logits | (N, 4672) |
| Value | (N, 1) |

The observation is always written from the point of view of the player to move:
planes 0-5 are that player's pieces and planes 6-11 the opponent's.

## Training

| Setting | Default |
|---|---|
| Parallel games | 2048 |
| Batches | 100000 |
| Frozen opponent refresh | every 500 batches |
| Discount factor | 0.99 |
| Entropy bonus | 0.01 |

### Self-play

- The trainee (the parameters that receive gradients) always occupies one
  player slot, the frozen opponent the other.
- PGX assigns colours to slots at random when an environment is created, so
  across a batch the trainee plays both colours. Nothing in the loss depends on
  a fixed colour.
- The frozen opponent is a copy of the trainee, refreshed every 500 batches,
  which keeps the opposition close to the current policy instead of letting it
  drift far ahead.

### Exploration

`epsilon` randomises the *opponent's* move only, which stops self-play from
collapsing onto a single line early on:

| Batch range | Epsilon |
|---|---|
| 0 – 1000 | 0.50 |
| 1000 – 3000 | 0.20 |
| 3000+ | 0.05 |

### Reward shaping

| Component | Formula |
|---|---|
| Game result | {-1, 0, 1} |
| Material gain | `clip(gain / 39, -1, 1) * 0.3` |
| Mobility penalty | `-(enemy_moves / 4672) * 0.2` |
| Draw while a piece up | -1.0 |
| Other draw | -0.5 |
| Step penalty | -0.001 |

`gain` is the mover's material difference (own minus opponent) after the move
minus the same difference before it, so capturing a queen is worth `9 / 39`.
The TD target is clipped to [-5, 5].

### Loss

```
actor_loss   = -mean(log_prob * stop_gradient(advantage) * trainee_moved)
critic_loss  = 0.5 * mean(advantage² * trainee_moved)
entropy_loss = -0.01 * mean(entropy * trainee_moved)
total_loss   = actor_loss + critic_loss + entropy_loss
```

### Recorded run

The published checkpoint came from a Kaggle T4 run of 70k+ batches. Win/draw
ratio over training (decisive results per draw):

| Batch | Win/draw ratio |
|---|---|
| 17k | ~1.00 |
| 24k | ~1.05 |
| 26k | ~1.08 |
| 72k+ | ~13.0 |

The model learns to prefer decisive results over draws as training progresses.
Those numbers come from the original single file script; this package keeps the
same objective and hyperparameters.

## Quick start

```bash
git clone https://github.com/alcsel/Recursive-Learning-Chess.git
cd Recursive-Learning-Chess
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For a GPU install, replace JAX with the CUDA build first:

```bash
pip install "jax[cuda12]"
```

Then train:

```bash
python -m chessrl train --num-envs 2048 --batches 100000
python -m chessrl train --resume-from checkpoints/chess_brain.msgpack
```

A short smoke run finishes in a minute on CPU:

```bash
python -m chessrl train --num-envs 8 --batches 4 --checkpoint /tmp/smoke.msgpack
```

Checkpoints are written every 25 batches to `checkpoints/chess_brain.msgpack`
unless `--checkpoint` says otherwise, and Ctrl+C saves before exiting.

## Playing against the model

```bash
python -m chessrl play --checkpoint checkpoints/chess_brain.msgpack
```

- Type moves in UCI notation, for example `e2e4`, `g8f6`, `e7e8q` for
  promotions.
- `help` prints the board, `quit` (or Ctrl+D) stops the game.
- The model plays White by default; use `--model-colour black` to switch, and
  `--no-board` to keep the output terse.
- The model's move is printed with the value head estimate for the position.

Older checkpoint files such as `satranc_beyni_jax 112k.msgpack` still load, they
just need to be passed explicitly:

```bash
python -m chessrl play --checkpoint "satranc_beyni_jax 112k.msgpack"
```

## Project layout

```
chessrl/
  config.py      constants, epsilon schedule, TrainingConfig, PlayConfig
  errors.py      exception hierarchy (no heavy imports)
  model.py       residual block and policy/value network
  board.py       PGX environment access, batching and UCI translation
  rewards.py     material balance, reward shaping, TD target
  checkpoint.py  reading and writing parameter trees
  training.py    loss, jitted update step and the self-play loop
  play.py        interactive human versus model game
  cli.py         `python -m chessrl train|play`
tests/           pytest suite; the JAX dependent tests skip when JAX is absent
```

Everything is reachable from the command line, so no module needs to be edited
to change a setting.

## Correctness notes

The refactor fixed two defects in the original scripts that are worth calling
out, because both silently corrupted the training signal:

1. **Material shaping used the wrong point of view.** The observation belongs to
   the player to move, but the next observation belongs to the *opponent*. The
   original compared the two directly, which doubled the term and, because PGX
   randomises colours per environment, inverted its sign for half of the
   parallel games. Both sides are now measured from the mover's point of view.
2. **Colour was confused with player slot.** `pgx.make("chess")` gives each
   environment a random `_player_order`, so "player 1" is White in some games
   and Black in others. The play script assumed it was always White, which
   mislabelled the sides in half of all games, and the material shaping guessed
   a colour mapping that did not hold at all.

Also fixed: a hardcoded absolute Windows checkpoint path, a hardcoded Kaggle
dataset path, a crash when the checkpoint directory was empty, `os.makedirs` on
a bare filename, and `exit()` instead of a clean shutdown.

## Tests

```bash
python -m pytest
```

The suite covers the epsilon schedule, configuration validation, the command
line interface, the reward formulas, the UCI translation (including all four
promotions) and a full training step. Tests that need JAX, Flax, Optax, PGX or
python-chess are skipped automatically when those packages are missing, so the
repository stays usable as a plain checkout.

## License

[MIT](LICENSE) © 2026 Selim Calici
