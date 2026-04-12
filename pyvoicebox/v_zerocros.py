"""V_ZEROCROS - Find zero crossings in a signal."""

from __future__ import annotations
import numpy as np


def v_zerocros(y, m='b', x=None) -> tuple[np.ndarray, np.ndarray]:
    """Find zero crossings in a signal.

    Parameters
    ----------
    y : array_like
        Input waveform.
    m : str, optional
        Mode string:
          'p' - positive crossings only
          'n' - negative crossings only
          'b' - both (default)
          'r' - round to sample values
    x : array_like, optional
        X-axis values for y. Default: 1-based indices (MATLAB convention).

    Returns
    -------
    t : ndarray
        X-axis positions of zero crossings (1-based if x not given).
    s : ndarray
        Estimated slope of y at the zero crossings.
    """
    y = np.asarray(y, dtype=float).ravel()
    s_sign = (y >= 0).astype(int)
    k = s_sign[1:] - s_sign[:-1]

    if 'p' in m:
        f = np.where(k > 0)[0]
    elif 'n' in m:
        f = np.where(k < 0)[0]
    else:
        f = np.where(k != 0)[0]

    s = y[f + 1] - y[f]
    # t uses 1-based indexing by default (MATLAB convention)
    t = (f + 1) - y[f] / s  # f is 0-based, add 1 for MATLAB convention

    if 'r' in m:
        t = np.round(t)

    if x is not None:
        x = np.asarray(x, dtype=float).ravel()
        tf = t - (f + 1)  # fractional sample
        t = x[f] * (1 - tf) + x[f + 1] * tf
        s = s / (x[f + 1] - x[f])

    return t, s
