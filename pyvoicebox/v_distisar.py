"""V_DISTISAR - Itakura-Saito distance between AR coefficients."""

import numpy as np
from pyvoicebox.v_lpcar2rr import v_lpcar2rr
from pyvoicebox.v_lpcar2ra import v_lpcar2ra


def v_distisar(ar1, ar2, mode=''):
    """Calculate the Itakura-Saito distance between AR coefficients.

    Parameters
    ----------
    ar1 : array_like, shape (nf1, p+1)
        AR coefficient sets.
    ar2 : array_like, shape (nf2, p+1)
        AR coefficient sets.
    mode : str, optional
        'x' for full distance matrix, 'd' for diagonal only.

    Returns
    -------
    d : ndarray
        Distance values.
    """
    ar1 = np.atleast_2d(np.asarray(ar1, dtype=float))
    ar2 = np.atleast_2d(np.asarray(ar2, dtype=float))
    nf1 = ar1.shape[0]
    nf2 = ar2.shape[0]

    m2 = v_lpcar2ra(ar2)
    m2[:, 0] *= 0.5

    if 'd' in mode or ('x' not in mode and nf1 == nf2):
        nx = min(nf1, nf2)
        d = 2.0 * np.sum(v_lpcar2rr(ar1[:nx, :]) * m2[:nx, :], axis=1) - np.log((ar2[:nx, 0] / ar1[:nx, 0]) ** 2) - 1.0
    else:
        d = 2.0 * v_lpcar2rr(ar1) @ m2.T - np.log((ar1[:, 0:1] ** (-1) * ar2[:, 0:1].T) ** 2) - 1.0
    return d
