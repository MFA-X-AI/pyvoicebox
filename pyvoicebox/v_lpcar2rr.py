"""V_LPCAR2RR - Convert autoregressive coefficients to autocorrelation coefficients."""

from __future__ import annotations
import numpy as np
from pyvoicebox.v_lpcar2rf import v_lpcar2rf
from pyvoicebox.v_lpcrf2rr import v_lpcrf2rr


def v_lpcar2rr(ar, p=None) -> np.ndarray:
    """Convert autoregressive coefficients to autocorrelation coefficients.

    Parameters
    ----------
    ar : array_like, shape (nf, n+1)
        Autoregressive coefficients including 0th coefficient.
    p : int, optional
        Number of output coefficients. Default is n.

    Returns
    -------
    rr : ndarray, shape (nf, p+1)
        Autocorrelation coefficients including 0th order coefficient.
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    k = ar[:, 0] ** (-2)
    if ar.shape[1] == 1:
        return k.reshape(-1, 1)

    if p is not None:
        rr, _ = v_lpcrf2rr(v_lpcar2rf(ar), p)
        return rr * k[:, np.newaxis]
    else:
        rr, _ = v_lpcrf2rr(v_lpcar2rf(ar))
        return rr * k[:, np.newaxis]
