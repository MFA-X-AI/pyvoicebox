"""V_DISTISPF - Itakura-Saito distance between power spectra."""

from __future__ import annotations
import numpy as np


def v_distispf(pf1, pf2, mode='') -> np.ndarray:
    """Calculate the Itakura-Saito spectral distance between power spectra.

    Parameters
    ----------
    pf1 : array_like, shape (nf1, p+1)
        Power spectra (DC to Nyquist).
    pf2 : array_like, shape (nf2, p+1)
        Power spectra (DC to Nyquist).
    mode : str, optional
        'x' for full distance matrix, 'd' for diagonal only.

    Returns
    -------
    d : ndarray
        Distance values.
    """
    pf1 = np.atleast_2d(np.asarray(pf1, dtype=float))
    pf2 = np.atleast_2d(np.asarray(pf2, dtype=float))
    nf1, p2 = pf1.shape
    p1 = p2 - 1
    nf2 = pf2.shape[0]

    if 'd' in mode or ('x' not in mode and nf1 == nf2):
        nx = min(nf1, nf2)
        r = pf1[:nx, :] / pf2[:nx, :]
        q = r - np.log(r)
        d = (np.sum(q[:, 1:p1], axis=1) + 0.5 * (q[:, 0] + q[:, p1])) / p1 - 1.0
    else:
        r = pf1[:, np.newaxis, :] / pf2[np.newaxis, :, :]
        q = r - np.log(r)
        d = (np.sum(q[:, :, 1:p1], axis=2) + 0.5 * (q[:, :, 0] + q[:, :, p1])) / p1 - 1.0
    return d
