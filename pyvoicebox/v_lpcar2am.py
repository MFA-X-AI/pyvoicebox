"""V_LPCAR2AM - Convert AR coefficients to AR coefficient matrix."""

from __future__ import annotations
import numpy as np


def v_lpcar2am(ar, p=None) -> tuple[np.ndarray, np.ndarray]:
    """Convert AR coefficients to AR coefficient matrix.

    Parameters
    ----------
    ar : array_like, shape (nf, p0+1)
        Autoregressive coefficients.
    p : int, optional
        Output order. Default is p0.

    Returns
    -------
    am : ndarray, shape (p+1, p+1, nf)
        AR coefficient matrix (upper triangular with 1s on diagonal).
    em : ndarray, shape (nf, p+1)
        Residual energy for each order.
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    if np.any(ar[:, 0] != 1):
        ar = ar / ar[:, 0:1]
    p0 = p1 - 1
    if p is None:
        p = p0

    am = np.zeros((nf, p + 1, p + 1))
    em = np.ones((nf, p + 1))
    e = np.ones((nf,))
    rf = ar.copy()

    jj = 0
    if p >= p0:
        for jj_idx in range(p + 1 - p0):
            am[:, jj_idx:jj_idx + p0 + 1, jj_idx] = ar
        jj = p + 1 - p0
    else:
        for j in range(p0, p + 1, -1):
            k = rf[:, j]
            d = (1 - k ** 2) ** (-1)
            e = e * d
            rf[:, 1:j] = (rf[:, 1:j] - k[:, np.newaxis] * rf[:, j-1:0:-1]) * d[:, np.newaxis]
        jj = 0

    for jj_idx in range(jj, p):
        j = p + 1 - jj_idx
        k = rf[:, j]
        d = (1 - k ** 2) ** (-1)
        e = e * d
        rf[:, 1:j] = (rf[:, 1:j] - k[:, np.newaxis] * rf[:, j-1:0:-1]) * d[:, np.newaxis]
        am[:, jj_idx:, jj_idx] = rf[:, :j]
        em[:, jj_idx] = e

    em[:, -1] = e / (1 - rf[:, 1] ** 2)
    am[:, -1, -1] = 1.0

    # Permute to (p+1, p+1, nf)
    am = np.transpose(am, (2, 1, 0))

    return am, em
