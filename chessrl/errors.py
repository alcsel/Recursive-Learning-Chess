"""Exceptions raised by this package.

Kept free of third party imports so callers can catch them without loading JAX.
"""

from __future__ import annotations

__all__ = ["CheckpointError", "ChessRLError"]


class ChessRLError(RuntimeError):
    """Base class for every error this package raises deliberately."""


class CheckpointError(ChessRLError):
    """Raised when a checkpoint cannot be read, written or matched."""
