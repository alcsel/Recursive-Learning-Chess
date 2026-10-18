from __future__ import annotations

from pathlib import Path

import pytest

from chessrl.cli import build_parser, main
from chessrl.config import BLACK, WHITE


class TestParser:
    def test_requires_a_command(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_train_keeps_the_documented_defaults(self):
        args = build_parser().parse_args(["train"])
        assert args.num_envs == 2048
        assert args.batches == 100_000
        assert args.resume_from is None
        assert args.checkpoint == Path("checkpoints") / "chess_brain.msgpack"

    def test_play_defaults_to_the_model_as_white(self):
        args = build_parser().parse_args(["play"])
        assert args.model_colour == "white"
        assert args.no_board is False

    def test_play_accepts_black(self):
        assert build_parser().parse_args(["play", "--model-colour", "black"]).model_colour == "black"

    def test_play_rejects_an_unknown_colour(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["play", "--model-colour", "green"])

    def test_colour_choices_match_the_constants(self):
        from chessrl.cli import _COLOURS

        assert _COLOURS == {"white": WHITE, "black": BLACK}


class TestPlayWithoutAHeavyStack:
    """The CLI surfaces a readable error before JAX is ever imported."""

    def test_missing_checkpoint_reports_an_error(self, capsys, monkeypatch):
        pytest.importorskip("flax")

        code = main(["play", "--checkpoint", "definitely-missing.msgpack"])

        assert code == 1
        assert "checkpoint not found" in capsys.readouterr().err
