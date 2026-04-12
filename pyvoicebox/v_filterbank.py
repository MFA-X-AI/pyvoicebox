"""V_FILTERBANK - Apply a bank of IIR filters to a signal."""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter


def v_filterbank(b, a, x, zi=None) -> tuple[np.ndarray, np.ndarray]:
    """Apply a bank of filters to a signal.

    Parameters
    ----------
    b : list of array_like or ndarray
        Numerator coefficients. If 2D, each row is a filter.
    a : list of array_like or ndarray
        Denominator coefficients. If 2D, each row is a filter.
    x : array_like
        Input signal.
    zi : list of array_like, optional
        Initial filter states.

    Returns
    -------
    y : ndarray
        Output signals, one column per filter.
    zf : list of ndarray
        Final filter states.
    """
    x = np.asarray(x, dtype=float).ravel()

    if isinstance(b, np.ndarray) and b.ndim == 2:
        nf = b.shape[0]
        b_list = [b[i, :] for i in range(nf)]
        a_list = [a[i, :] for i in range(nf)]
    elif isinstance(b, list):
        nf = len(b)
        b_list = b
        a_list = a
    else:
        nf = 1
        b_list = [np.asarray(b, dtype=float).ravel()]
        a_list = [np.asarray(a, dtype=float).ravel()]

    y = np.zeros((len(x), nf))
    zf = []

    for i in range(nf):
        bi = np.asarray(b_list[i], dtype=float).ravel()
        ai = np.asarray(a_list[i], dtype=float).ravel()
        if zi is not None and i < len(zi):
            yi, zfi = lfilter(bi, ai, x, zi=zi[i])
        else:
            yi, zfi = lfilter(bi, ai, x, zi=None)
            if zfi is None:
                zfi = np.zeros(max(len(bi), len(ai)) - 1)
        y[:, i] = yi
        zf.append(zfi)

    return y, zf
