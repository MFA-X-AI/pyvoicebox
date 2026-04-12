"""V_LPCRAND - Generate random stable polynomials."""

from __future__ import annotations
import numpy as np
from pyvoicebox.v_lpcrf2ar import v_lpcrf2ar


def v_lpcrand(p, n=1, bw=0) -> np.ndarray:
    """Generate random stable polynomials.

    Parameters
    ----------
    p : int
        Polynomial order.
    n : int, optional
        Number of polynomials. Default is 1.
    bw : float or array_like, optional
        Minimum pole bandwidth as fraction of sampling frequency. Default is 0.

    Returns
    -------
    ar : ndarray, shape (n, p+1)
        Random stable AR polynomials.
    """
    if p == 0:
        return np.ones((n, 1))

    bw = np.asarray(bw, dtype=float)
    if bw.ndim == 0:
        bw = float(bw)

    if bw == 0 if np.isscalar(bw) else not np.any(bw):
        ar = v_lpcrf2ar(2 * np.random.rand(n, p + 1) - 1)
    else:
        bw_arr = np.atleast_1d(bw)
        k = np.exp(-np.pi * bw_arr.reshape(-1, 1) * np.arange(p + 1))
        if k.shape[0] == 1:
            ar = v_lpcrf2ar(2 * np.random.rand(n, p + 1) - 1) * np.tile(k, (n, 1))
        else:
            ar = v_lpcrf2ar(2 * np.random.rand(n, p + 1) - 1) * k

    return ar
