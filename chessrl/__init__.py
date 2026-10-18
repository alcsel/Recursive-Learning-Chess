"""Chess reinforcement learning agent built on JAX, Flax and PGX.

The package is split so that the expensive dependencies are only imported by
the modules that need them: :mod:`chessrl.config` and :mod:`chessrl.cli` work
without JAX, while the model, training and play modules require the full
machine learning stack.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
