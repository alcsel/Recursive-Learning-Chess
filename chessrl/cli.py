"""Command line interface: ``python -m chessrl train|play``.

Only :mod:`chessrl.config` and :mod:`chessrl.errors` are imported at module
level, so ``--help`` works on a machine without JAX installed. The heavy
modules are imported inside the handlers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    BLACK,
    DEFAULT_CHECKPOINT,
    DEFAULT_FROZEN_UPDATE_INTERVAL,
    DEFAULT_NUM_ENVS,
    DEFAULT_SEED,
    WHITE,
    PlayConfig,
    TrainingConfig,
    colour_name,
)
from .errors import ChessRLError

__all__ = ["build_parser", "main"]

_COLOURS = {"white": WHITE, "black": BLACK}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chessrl",
        description="Train and play with the JAX chess reinforcement learning agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="run self-play training")
    train.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS, help="parallel games")
    train.add_argument("--batches", type=int, default=TrainingConfig.batches, help="update steps")
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="where the trained weights are written",
    )
    train.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="continue from an existing checkpoint instead of starting from scratch",
    )
    train.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    train.add_argument("--gamma", type=float, default=TrainingConfig.gamma)
    train.add_argument("--entropy-coefficient", type=float, default=TrainingConfig.entropy_coefficient)
    train.add_argument(
        "--frozen-update-interval",
        type=int,
        default=DEFAULT_FROZEN_UPDATE_INTERVAL,
        help="batches between frozen opponent refreshes",
    )
    train.add_argument("--save-every", type=int, default=TrainingConfig.save_every)
    train.add_argument("--log-every", type=int, default=TrainingConfig.log_every)

    play = subparsers.add_parser("play", help="play against a trained checkpoint")
    play.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    play.add_argument(
        "--model-colour",
        choices=sorted(_COLOURS),
        default="white",
        help="the colour the model plays",
    )
    play.add_argument("--seed", type=int, default=DEFAULT_SEED)
    play.add_argument("--no-board", action="store_true", help="hide the board after every move")

    return parser


def _run_train(args: argparse.Namespace) -> int:
    from .training import run_training

    config = TrainingConfig(
        num_envs=args.num_envs,
        batches=args.batches,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        entropy_coefficient=args.entropy_coefficient,
        frozen_update_interval=args.frozen_update_interval,
        save_every=args.save_every,
        log_every=args.log_every,
        checkpoint=args.checkpoint,
        resume_from=args.resume_from,
    )
    run_training(config)
    return 0


def _run_play(args: argparse.Namespace) -> int:
    from .play import play

    config = PlayConfig(
        checkpoint=args.checkpoint,
        model_colour=_COLOURS[args.model_colour],
        seed=args.seed,
        show_board=not args.no_board,
    )
    print(f"The model plays {colour_name(config.model_colour)}.")
    play(config)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"train": _run_train, "play": _run_play}

    try:
        return handlers[args.command](args)
    except ChessRLError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
