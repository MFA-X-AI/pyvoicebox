"""MATLAB compatibility helpers for VOICEBOX Python port."""

import numpy as np


def atleast_col(x):
    """Ensure x is at least a 2D column vector."""
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def atleast_row(x):
    """Ensure x is at least a 2D row vector."""
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


def matlab_reshape(x, shape):
    """Reshape using MATLAB column-major (Fortran) order."""
    return np.reshape(x, shape, order='F')


def first_nonsingleton(x):
    """Return the index of the first non-singleton dimension (0-based), or 0."""
    for i, s in enumerate(x.shape):
        if s > 1:
            return i
    return 0
