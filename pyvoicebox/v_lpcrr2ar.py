"""V_LPCRR2AR - Convert autocorrelation coefficients to AR coefficients."""

from __future__ import annotations
import numpy as np


def v_lpcrr2ar(rr) -> tuple[np.ndarray, np.ndarray]:
    """Convert autocorrelation coefficients to AR coefficients.

    Parameters
    ----------
    rr : array_like, shape (nf, p+1)
        Autocorrelation coefficients.

    Returns
    -------
    ar : ndarray, shape (nf, p+1)
        AR coefficients.
    e : ndarray, shape (nf,)
        Residual energy.
    """
    rr = np.atleast_2d(np.asarray(rr, dtype=float))
    nf, p1 = rr.shape
    p = p1 - 1

    ar = np.ones((nf, p1))
    ar[:, 1] = -rr[:, 1] / rr[:, 0]
    e = rr[:, 0] * (ar[:, 1] ** 2 - 1)

    for n in range(2, p + 1):
        k = (rr[:, n] + np.sum(rr[:, n-1:0:-1] * ar[:, 1:n], axis=1)) / e
        ar[:, 1:n] = ar[:, 1:n] + k[:, np.newaxis] * ar[:, n-1:0:-1]
        ar[:, n] = k
        e = e * (1 - k ** 2)

    e = -e
    return ar, e
