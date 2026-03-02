"""V_MODSYM - Symmetric modulus function."""

import numpy as np


def v_modsym(x, y=1, r=None):
    """Symmetric modulus function.

    Adds an integer multiple of y onto x so that it lies in the range
    [r-y/2, r+y/2) if y is positive or (r-y/2, r+y/2] if y is negative.

    Parameters
    ----------
    x : array_like
        Input data.
    y : float or array_like, optional
        Modulus. Default is 1.
    r : float or array_like, optional
        Reference data. Default is 0.

    Returns
    -------
    z : ndarray
        Output data.
    k : ndarray
        Integer multiple of y that was added.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if r is None:
        v = 0.5 * y
    else:
        r = np.asarray(r, dtype=float)
        v = 0.5 * y - r
    z = np.mod(x + v, y) - v
    k = np.round((z - x) / y).astype(int)
    return z, k
