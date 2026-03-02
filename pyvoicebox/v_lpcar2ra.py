"""V_LPCAR2RA - Convert AR filter to inverse filter autocorrelation coefficients."""

import numpy as np


def v_lpcar2ra(ar):
    """Convert AR filter to inverse filter autocorrelation coefficients.

    Parameters
    ----------
    ar : array_like, shape (nf, p+1)
        Autoregressive coefficients.

    Returns
    -------
    ra : ndarray, shape (nf, p+1)
        Inverse filter autocorrelation coefficients.
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    ra = np.zeros((nf, p1))
    for i in range(p1):
        ra[:, i] = np.sum(ar[:, :p1 - i] * ar[:, i:], axis=1)
    return ra
