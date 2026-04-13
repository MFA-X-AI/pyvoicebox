"""MATLAB compatibility helpers for VOICEBOX Python port."""

from __future__ import annotations

import numpy as np


def _require_matplotlib(func_name):
    """Import matplotlib.pyplot with a friendly error if it isn't installed."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            f"{func_name} requires matplotlib, which is an optional dependency. "
            f"Install it with: pip install 'py-voicebox[plot]'"
        ) from e


def atleast_col(x) -> np.ndarray:
    """Ensure x is at least a 2D column vector."""
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def atleast_row(x) -> np.ndarray:
    """Ensure x is at least a 2D row vector."""
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


def matlab_reshape(x, shape) -> np.ndarray:
    """Reshape using MATLAB column-major (Fortran) order."""
    return np.reshape(x, shape, order='F')


def first_nonsingleton(x) -> int:
    """Return the index of the first non-singleton dimension (0-based), or 0."""
    for i, s in enumerate(x.shape):
        if s > 1:
            return i
    return 0
