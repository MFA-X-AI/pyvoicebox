"""V_EWGRPDEL - Energy-weighted group delay waveform."""

import numpy as np
from scipy.signal.windows import hamming
from scipy.signal import lfilter


def v_ewgrpdel(x, w=None, m=None):
    """Calculate energy-weighted group delay waveform.

    Parameters
    ----------
    x : array_like
        Input signal.
    w : int or array_like, optional
        Window or window length (default: Hamming of length(x)).
    m : int, optional
        Center sample of window (1-based, default: (1+len(w))/2).

    Returns
    -------
    y : ndarray
        Energy-weighted group delay waveform.
    mm : int
        Actual value of m used.
    """
    x = np.asarray(x, dtype=float).ravel()

    if w is None:
        w = hamming(len(x), sym=True)
    elif np.isscalar(w):
        w = hamming(int(w), sym=True)
    else:
        w = np.asarray(w, dtype=float).ravel()

    w = w ** 2
    lw = len(w)

    if m is None:
        m = (1 + lw) / 2.0
    m = max(int(round(m)), 1)
    mm = m

    wn = w * (m - np.arange(1, lw + 1))
    x2 = np.concatenate([x, np.zeros(m - 1)]) ** 2
    yn = lfilter(wn, [1.0], x2)
    yd = lfilter(w, [1.0], x2)
    yd[yd < np.finfo(float).eps] = 1.0
    y = yn[m - 1:] / yd[m - 1:]
    return y, mm
