"""V_LPCZZ2AR - Convert z-plane poles to AR coefficients."""

import numpy as np


def v_lpczz2ar(zz):
    """Convert z-plane poles to AR coefficients.

    Parameters
    ----------
    zz : array_like, shape (nf, p)
        Z-plane poles.

    Returns
    -------
    ar : ndarray, shape (nf, p+1)
        AR coefficients.
    """
    zz = np.atleast_2d(np.asarray(zz, dtype=complex))
    nf, p = zz.shape
    ar = np.zeros((nf, p + 1))
    for k in range(nf):
        ar[k, :] = np.real(np.poly(zz[k, :]))
    return ar
