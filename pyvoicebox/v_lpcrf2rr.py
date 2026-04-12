"""V_LPCRF2RR - Convert reflection coefficients to autocorrelation coefficients."""

from __future__ import annotations
import numpy as np


def v_lpcrf2rr(rf, p=None) -> tuple[np.ndarray, np.ndarray]:
    """Convert reflection coefficients to autocorrelation coefficients.

    Parameters
    ----------
    rf : array_like, shape (nf, n+1)
        Reflection coefficients, one row per frame.
    p : int, optional
        Number of autocorrelation coefficients to calculate. Default is n.

    Returns
    -------
    rr : ndarray, shape (nf, p+1)
        Autocorrelation coefficients.
    ar : ndarray, shape (nf, n+1)
        AR filter coefficients.
    """
    rf = np.atleast_2d(np.asarray(rf, dtype=float))
    nf, p1 = rf.shape
    p0 = p1 - 1

    if p0 == 0:
        rr = np.ones((nf, 1))
        ar = np.ones((nf, 1))
        return rr, ar

    a = rf[:, 1:2].copy()  # shape (nf, 1)
    rr = np.zeros((nf, p1))
    rr[:, 0] = 1.0
    rr[:, 1] = -a[:, 0]
    e = a[:, 0] ** 2 - 1.0

    for n in range(2, p0 + 1):
        k = rf[:, n]
        rr[:, n] = k * e - np.sum(rr[:, n-1:0:-1] * a, axis=1)
        new_col = k.copy()
        a = np.column_stack([a + k[:, np.newaxis] * a[:, n-2::-1], new_col])
        e = e * (1 - k ** 2)

    ar = np.column_stack([np.ones((nf, 1)), a])
    r0 = np.sum(rr * ar, axis=1) ** (-1)
    rr = rr * r0[:, np.newaxis]

    if p is not None:
        if p < p0:
            rr = rr[:, :p + 1]
        elif p > p0:
            rr = np.column_stack([rr, np.zeros((nf, p - p0))])
            af = -ar[:, p1-1:0:-1]
            for i in range(p0 + 1, p + 1):
                rr[:, i] = np.sum(af * rr[:, i - p0:i], axis=1)

    return rr, ar
