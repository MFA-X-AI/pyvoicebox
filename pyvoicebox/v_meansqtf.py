"""V_MEANSQTF - Mean square transfer function of a filter."""

import numpy as np
from pyvoicebox.v_lpcar2rr import v_lpcar2rr
from pyvoicebox.v_lpcar2ra import v_lpcar2ra


def v_meansqtf(b, a=None):
    """Calculate the mean square transfer function of a filter.

    This equals the average output power when the filter is fed
    with unit variance white noise.

    Parameters
    ----------
    b : array_like
        Numerator filter coefficients.
    a : array_like, optional
        Denominator filter coefficients. Default is [1].

    Returns
    -------
    d : float
        Mean square transfer function.
    """
    b = np.asarray(b, dtype=float).ravel()
    if a is None or len(np.atleast_1d(a)) == 1:
        return float(b @ b)

    a = np.asarray(a, dtype=float).ravel()
    m = v_lpcar2ra(b.reshape(1, -1))
    m[0, 0] *= 0.5
    rr = v_lpcar2rr(a.reshape(1, -1), len(m[0]) - 1)
    d = float(2.0 * rr @ m.T)
    return d
