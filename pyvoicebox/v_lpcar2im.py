"""V_LPCAR2IM - Convert AR coefficients to impulse response."""

import numpy as np
from scipy.signal import lfilter


def v_lpcar2im(ar, np_out=None):
    """Convert AR coefficients to impulse response.

    Parameters
    ----------
    ar : array_like, shape (nf, p+1)
        AR coefficients, one frame per row.
    np_out : int, optional
        Number of impulse response samples minus 1. Default is p.

    Returns
    -------
    im : ndarray, shape (nf, np_out+1)
        Impulse response.
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    if np_out is None:
        np_out = p1 - 1

    im = np.zeros((nf, np_out + 1))
    x = np.zeros(np_out + 1)
    x[0] = 1.0
    for k in range(nf):
        im[k, :] = lfilter([1.0], ar[k, :], x)
    return im
