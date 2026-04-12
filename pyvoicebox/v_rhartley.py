"""V_RHARTLEY - Calculate the Hartley transform of real data."""

from __future__ import annotations
import numpy as np


def v_rhartley(x, n=None) -> np.ndarray:
    """Calculate the Hartley transform of real data Y=(X,N).

    Data is truncated/padded to length N if specified.
    The inverse transformation is x = v_rhartley(y, n) / n.

    Parameters
    ----------
    x : array_like
        Real input data.
    n : int, optional
        Transform length. Default: length of x.

    Returns
    -------
    y : ndarray
        Hartley transform of x.
    """
    x = np.asarray(x, dtype=float)
    x = np.real(x)

    # MATLAB's fft operates on the first non-singleton dimension by default
    # (columns for 2-D arrays). Match this behavior.
    if x.ndim >= 2:
        axis = 0
    else:
        axis = -1

    if n is None:
        y = np.fft.fft(x, axis=axis)
    else:
        y = np.fft.fft(x, n=n, axis=axis)

    y = np.real(y) - np.imag(y)

    return y
