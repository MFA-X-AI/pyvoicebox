"""V_LPCBWEXP - Expand formant bandwidths of LPC filter."""

from __future__ import annotations
import numpy as np


def v_lpcbwexp(ar, bw) -> np.ndarray:
    """Expand formant bandwidths of LPC filter.

    Parameters
    ----------
    ar : array_like, shape (nf, p+1)
        Autoregressive coefficients.
    bw : float
        Bandwidth expansion factor. The radius of each pole is multiplied
        by exp(-bw*pi).

    Returns
    -------
    arx : ndarray, shape (nf, p+1)
        Bandwidth-expanded AR coefficients.
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    k = np.exp(-np.pi * np.arange(p1) * bw)
    arx = ar * k[np.newaxis, :]
    return arx
