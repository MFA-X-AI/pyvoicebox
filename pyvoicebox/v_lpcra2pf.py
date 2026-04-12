"""V_LPCRA2PF - Convert inverse filter autocorrelation to power spectrum."""

from __future__ import annotations
import numpy as np
from pyvoicebox.v_rfft import v_rfft


def v_lpcra2pf(ra, np_out=None) -> np.ndarray:
    """Convert inverse filter autocorrelation to power spectrum.

    Parameters
    ----------
    ra : array_like, shape (nf, p+1)
        Inverse filter autocorrelation coefficients.
    np_out : int, optional
        Number of output frequencies minus 1. Default is p.

    Returns
    -------
    pf : ndarray, shape (nf, np_out+2)
        Power spectrum.
    """
    ra = np.atleast_2d(np.asarray(ra, dtype=float))
    nf, p1 = ra.shape
    if np_out is None:
        np_out = p1 - 1
    pp = 2 * np_out + 2

    if pp >= 2 * p1:
        # Zero-pad and mirror
        data = np.column_stack([ra, np.zeros((nf, pp - 2 * p1 + 1)), ra[:, p1 - 1:0:-1]])
        pf = np.abs(v_rfft(data.T).T) ** (-1)
    else:
        data = np.column_stack([ra[:, :np_out + 2], ra[:, np_out:0:-1]])
        pf = np.abs(v_rfft(data.T).T) ** (-1)

    return pf
