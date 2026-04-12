"""V_LPCPP2CW - Convert power spectrum polynomial to power spectrum zeros."""

from __future__ import annotations
import numpy as np


def v_lpcpp2cw(pp) -> np.ndarray:
    """Convert power spectrum polynomial in cos(w) to power spectrum zeros.

    Parameters
    ----------
    pp : array_like, shape (nf, p+1)
        Power spectrum polynomial coefficients.

    Returns
    -------
    cw : ndarray, shape (nf, p)
        Power spectrum zeros (roots of the polynomial).
    """
    pp = np.atleast_2d(np.asarray(pp, dtype=complex))
    nf, p1 = pp.shape
    cw = np.zeros((nf, p1 - 1), dtype=complex)
    for k in range(nf):
        cw[k, :] = np.roots(pp[k, :])
    return cw
