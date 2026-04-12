"""V_RFFT - Calculate the DFT of real data, returning only the first half."""

from __future__ import annotations
import numpy as np


def v_rfft(x, n=None, d=None) -> np.ndarray:
    """Calculate the DFT of real data Y=(X,N,D).

    Data is truncated/padded to length N if specified.
      N even: (N+2)/2 points are returned with the first and last being real
      N odd:  (N+1)/2 points are returned with the first being real
    In all cases floor(1+N/2) points are returned.
    D is the dimension (0-based axis) along which to do the DFT.

    Parameters
    ----------
    x : array_like
        Real input data.
    n : int, optional
        Transform length. Default is the size of x along axis d.
    d : int, optional
        Axis along which to compute the FFT. Default is first axis with size > 1.

    Returns
    -------
    y : ndarray
        The first floor(1+N/2) points of the DFT.
    """
    x = np.asarray(x, dtype=float)
    s = x.shape

    # Scalar case
    if x.size == 1:
        return x.copy()

    if d is None:
        # Find first non-singleton dimension
        for i, si in enumerate(s):
            if si > 1:
                d = i
                break
        if d is None:
            d = 0
        if n is None:
            n = s[d]

    if n is None:
        n = s[d]

    y = np.fft.fft(x, n=n, axis=d)

    # Keep only first floor(1+n/2) points along axis d
    keep = 1 + n // 2
    slices = [slice(None)] * y.ndim
    slices[d] = slice(0, keep)
    y = y[tuple(slices)]

    return y
