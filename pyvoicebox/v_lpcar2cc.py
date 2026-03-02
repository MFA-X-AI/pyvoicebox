"""V_LPCAR2CC - Convert AR filter to complex cepstrum."""

import numpy as np
from scipy.signal import lfilter


def v_lpcar2cc(ar, np_out=None):
    """Convert AR filter to complex cepstrum.

    Parameters
    ----------
    ar : array_like, shape (nf, n+1)
        AR coefficients, one frame per row.
    np_out : int, optional
        Number of cepstral coefficients to calculate. Default is n.

    Returns
    -------
    cc : ndarray, shape (nf, np_out)
        Complex cepstral coefficients, excluding c(0).
    c0 : ndarray, shape (nf, 1)
        Coefficient c(0).
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    p = p1 - 1
    if np_out is None:
        np_out = p

    cc = np.zeros((nf, np_out))
    if np.any(ar[:, 0] != 1):
        c0 = -np.log(ar[:, 0]).reshape(-1, 1)
        ar = ar / ar[:, 0:1]
    else:
        c0 = np.zeros((nf, 1))

    cm = 1.0 / np.arange(1, np_out + 1)
    if np_out > p:
        xm = -np.arange(1, p + 1, dtype=float)
        nz = np_out - p
        for k in range(nf):
            inp = np.concatenate([ar[k, 1:p1] * xm, np.zeros(nz)])
            cc[k, :] = lfilter([1.0], ar[k, :], inp) * cm
    else:
        p1_out = np_out + 1
        xm = -np.arange(1, np_out + 1, dtype=float)
        for k in range(nf):
            cc[k, :] = lfilter([1.0], ar[k, :], ar[k, 1:p1_out] * xm) * cm

    return cc, c0
