"""V_LPCPF2RR - Convert power spectrum to autocorrelation coefficients."""

import numpy as np
from pyvoicebox.v_irfft import v_irfft


def v_lpcpf2rr(pf, p=None):
    """Convert power spectrum to autocorrelation coefficients.

    Parameters
    ----------
    pf : array_like, shape (nf, p2)
        Power spectrum.
    p : int, optional
        Number of output coefficients minus 1. Default is p2-2.

    Returns
    -------
    rr : ndarray, shape (nf, p+1)
        Autocorrelation coefficients.
    """
    pf = np.atleast_2d(np.asarray(pf, dtype=float))
    nf, p2 = pf.shape
    if p is None:
        p = p2 - 2

    # MATLAB: ir = v_irfft(pf,[],2) -- irfft along rows (dim 2)
    # Transpose to columns, apply irfft along dim 0, transpose back
    ir = v_irfft(pf.T).T
    if p > p2 - 2:
        rr = np.column_stack([ir[:, :p2 - 1], np.zeros((nf, p + 2 - p2))])
    else:
        rr = ir[:, :p + 1]

    return rr
