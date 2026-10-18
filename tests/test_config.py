from __future__ import annotations

from pathlib import Path

import pytest

from chessrl.config import (
    BLACK,
    WHITE,
    PlayConfig,
    TrainingConfig,
    colour_name,
    epsilon_for,
)


class TestEpsilonSchedule:
    @pytest.mark.parametrize(
        ("batch", "expected"),
        [
            (1, 0.5),
            (999, 0.5),
            (1000, 0.2),
            (2999, 0.2),
            (3000, 0.05),
            (1_000_000, 0.05),
        ],
    )
    def test_follows_the_documented_steps(self, batch, expected):
        assert epsilon_for(batch) == pytest.approx(expected)

    def test_rejects_a_non_positive_batch(self):
        with pytest.raises(ValueError, match="positive"):
            epsilon_for(0)

    def test_accepts_a_custom_schedule(self):
        schedule = ((10, 0.9), (20, 0.4), (30, 0.1))
        assert epsilon_for(5, schedule) == 0.9
        assert epsilon_for(15, schedule) == 0.4
        assert epsilon_for(99, schedule) == 0.1


class TestTrainingConfig:
    def test_defaults_match_the_trained_run(self):
        config = TrainingConfig()
        assert (config.num_envs, config.gamma) == (2048, 0.99)
        assert config.frozen_update_interval == 500
        assert config.checkpoint == Path("checkpoints") / "chess_brain.msgpack"

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"num_envs": 0}, "num_envs"),
            ({"batches": 0}, "batches"),
            ({"gamma": 1.5}, "gamma"),
            ({"frozen_update_interval": 0}, "frozen_update_interval"),
            ({"save_every": 0}, "save_every"),
            ({"log_every": 0}, "log_every"),
        ],
    )
    def test_rejects_nonsensical_values(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            TrainingConfig(**overrides)

    def test_schedules_report_their_multiples(self):
        config = TrainingConfig(frozen_update_interval=500, save_every=25, log_every=10)
        assert config.should_update_opponent(1000)
        assert not config.should_update_opponent(1001)
        assert config.should_save(25)
        assert config.should_log(10)

    def test_training_starts_from_scratch_unless_told_otherwise(self):
        assert TrainingConfig().resume_from is None
        assert TrainingConfig(resume_from=Path("a.msgpack")).resume_from == Path("a.msgpack")


class TestPlayConfig:
    def test_the_human_takes_the_other_colour(self):
        assert PlayConfig(model_colour=WHITE).human_colour == BLACK
        assert PlayConfig(model_colour=BLACK).human_colour == WHITE

    def test_rejects_an_unknown_colour(self):
        with pytest.raises(ValueError, match="model_colour"):
            PlayConfig(model_colour=2)


def test_colour_name_is_readable():
    assert colour_name(WHITE) == "White"
    assert colour_name(BLACK) == "Black"
