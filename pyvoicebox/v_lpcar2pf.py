"""V_LPCAR2PF - Convert AR coefficients to power spectrum."""

from __future__ import annotations
import numpy as np
from pyvoicebox.v_rfft import v_rfft


def v_lpcar2pf(ar, np_out=None) -> tuple[np.ndarray, np.ndarray]:
    """Convert AR coefficients to power spectrum.

    Parameters
    ----------
    ar : array_like, shape (nf, n)
        AR coefficients, one frame per row.
    np_out : int, optional
        Size of output spectrum is np_out+1. Default is n-1.

    Returns
    -------
    pf : ndarray, shape (nf, np_out+1)
        Power spectrum from DC to Nyquist.
    f : ndarray, shape (np_out+1,)
        Normalized frequencies (0 to 0.5).
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    if np_out is None:
        np_out = p1 - 1

    pf = np.abs(v_rfft(ar.T, 2 * np_out).T) ** (-2)
    f = np.arange(np_out + 1) / (2 * np_out)

    return pf, f
