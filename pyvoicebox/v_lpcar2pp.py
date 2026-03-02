"""V_LPCAR2PP - Convert AR filter to power spectrum polynomial in cos(w)."""

import numpy as np
from pyvoicebox.v_lpcar2ra import v_lpcar2ra


def _chebyshev_matrix(p1):
    """Build the Chebyshev polynomial transform matrix of size p1 x p1."""
    p = p1 - 1
    tp = np.zeros((p1, p1))
    tp[0, p] = 2.0
    if p1 > 1:
        tp[1, p - 1] = 2.0
    for i in range(2, p1):
        # MATLAB 1-based: i_m = i+1, so p+2-i_m = p+1-i, p+3-i_m = p+2-i
        # tp(i_m, p+2-i_m:p) = 2*tp(i_m-1, p+3-i_m:p1) - tp(i_m-2, p+2-i_m:p)
        # 0-based target: columns p+1-(i+1) to p-1 = p-i to p-1
        # 0-based source1: columns p+2-(i+1) to p1-1 = p+1-i to p
        # 0-based source2: same as target
        col_start = p - i        # 0-based start of target range
        col_end = p              # 0-based exclusive end (p-1 inclusive)
        src_start = p + 1 - i    # 0-based start of source range
        src_end = p1             # 0-based exclusive end
        tp[i, col_start:col_end] = 2.0 * tp[i-1, src_start:src_end] - tp[i-2, col_start:col_end]
        tp[i, p] = -tp[i-2, p]  # tp(i_m, p1) = -tp(i_m-2, p1), p1 in 0-based = p
    tp[0, p] = 1.0
    return tp


def v_lpcar2pp(ar):
    """Convert AR filter to power spectrum polynomial in cos(w).

    Parameters
    ----------
    ar : array_like, shape (nf, p+1)
        Autoregressive coefficients.

    Returns
    -------
    pp : ndarray, shape (nf, p+1)
        Power spectrum polynomial coefficients in cos(w).
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    ra = v_lpcar2ra(ar)
    tp = _chebyshev_matrix(p1)
    pp = ra @ tp
    return pp
