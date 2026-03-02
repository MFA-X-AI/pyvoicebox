"""V_LPCAR2ZZ - Convert AR filter to z-plane poles."""

import numpy as np


def v_lpcar2zz(ar):
    """Convert AR filter to z-plane poles.

    Parameters
    ----------
    ar : array_like, shape (nf, p+1)
        Autoregressive coefficients.

    Returns
    -------
    zz : ndarray, shape (nf, p)
        Z-plane poles.
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    zz = np.zeros((nf, p1 - 1), dtype=complex)
    for k in range(nf):
        zz[k, :] = np.roots(ar[k, :])
    return zz
