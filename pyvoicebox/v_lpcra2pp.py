"""V_LPCRA2PP - Convert inverse filter autocorrelation to power spectrum polynomial."""

import numpy as np
from pyvoicebox.v_lpcar2pp import _chebyshev_matrix


def v_lpcra2pp(ra):
    """Convert inverse filter autocorrelation to power spectrum polynomial in cos(w).

    Parameters
    ----------
    ra : array_like, shape (nf, p+1)
        Inverse filter autocorrelation coefficients.

    Returns
    -------
    pp : ndarray, shape (nf, p+1)
        Power spectrum polynomial coefficients.
    """
    ra = np.atleast_2d(np.asarray(ra, dtype=float))
    nf, p1 = ra.shape
    tp = _chebyshev_matrix(p1)
    pp = ra @ tp
    return pp
