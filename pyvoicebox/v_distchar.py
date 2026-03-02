"""V_DISTCHAR - COSH spectral distance between AR coefficients."""

import numpy as np
from pyvoicebox.v_lpcar2rr import v_lpcar2rr
from pyvoicebox.v_lpcar2ra import v_lpcar2ra


def v_distchar(ar1, ar2, mode=''):
    """Calculate the COSH spectral distance between AR coefficients.

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
    nf1, p1 = ar1.shape
    nf2 = ar2.shape[0]
    p2 = p1 + 1

    m1 = np.zeros((nf1, 2 * p1))
    m2 = np.zeros((nf2, 2 * p1))
    m1[:, :p1] = v_lpcar2rr(ar1)
    m1[:, p1:] = v_lpcar2ra(ar1)
    m1[:, 0] *= 0.5
    m1[:, p1] *= 0.5
    m2[:, p1:] = v_lpcar2rr(ar2)
    m2[:, :p1] = v_lpcar2ra(ar2)

    if 'd' in mode or ('x' not in mode and nf1 == nf2):
        nx = min(nf1, nf2)
        d = np.sum(m1[:nx, :] * m2[:nx, :], axis=1) - 1.0
    else:
        d = m1 @ m2.T - 1.0
    return d
