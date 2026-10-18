"""Saving and loading the network parameters."""

from __future__ import annotations

from pathlib import Path

from flax import serialization

from .errors import CheckpointError

__all__ = ["CheckpointError", "load_params", "save_params"]


def save_params(params, path: str | Path) -> Path:
    """Write ``params`` to ``path``, creating parent directories as needed."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = serialization.to_bytes(params)
    try:
        target.write_bytes(payload)
    except OSError as error:
        raise CheckpointError(f"could not write the checkpoint to {target}: {error}") from error

    return target


def load_params(params, path: str | Path):
    """Restore a parameter tree, raising a readable error when it is unusable."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise CheckpointError(
            f"checkpoint not found: {source}. Train one with "
            "'python -m chessrl train' or point --checkpoint at an existing file."
        )

    try:
        return serialization.from_bytes(params, source.read_bytes())
    except Exception as error:  # flax raises a mix of ValueError and shape errors
        raise CheckpointError(
            f"{source} does not match the current ChessBrain architecture: {error}"
        ) from error
