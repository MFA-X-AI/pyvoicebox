"""V_LPCZZ2CC - Convert poles to complex cepstrum."""

from __future__ import annotations
import numpy as np


def v_lpczz2cc(zz, np_out=None) -> np.ndarray:
    """Convert poles to complex cepstrum.

    Parameters
    ----------
    zz : array_like, shape (nf, p)
        Z-plane poles.
    np_out : int, optional
        Number of cepstral coefficients. Default is p.

    Returns
    -------
    cc : ndarray, shape (nf, np_out)
        Complex cepstral coefficients.
    """
    zz = np.atleast_2d(np.asarray(zz, dtype=complex))
    nf, p = zz.shape
    if np_out is None:
        np_out = p

    cc = np.zeros((nf, np_out))
    yy = zz.T.copy()  # shape (p, nf)

    if p < 2:
        cc[:, 0] = np.real(zz).ravel()
        for k in range(1, np_out):
            yy = yy * zz.T
            cc[:, k] = np.real(yy).ravel() / (k + 1)
    else:
        cc[:, 0] = np.sum(np.real(yy), axis=0)
        for k in range(1, np_out):
            yy = yy * zz.T
            cc[:, k] = np.sum(np.real(yy), axis=0) / (k + 1)

    return cc
