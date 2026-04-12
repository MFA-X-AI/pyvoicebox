"""V_LPCAR2RF - Convert autoregressive coefficients to reflection coefficients."""

from __future__ import annotations
import numpy as np


def v_lpcar2rf(ar) -> np.ndarray:
    """Convert autoregressive coefficients to reflection coefficients.

    Parameters
    ----------
    ar : array_like, shape (nf, p+1)
        Autoregressive coefficients with ar[:, 0] = 1.

    Returns
    -------
    rf : ndarray, shape (nf, p+1)
        Reflection coefficients with rf[:, 0] = 1.
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    if p1 == 1:
        return np.ones((nf, 1))

    # Normalize so first coefficient is 1
    if np.any(ar[:, 0] != 1):
        ar = ar / ar[:, 0:1]

    rf = ar.copy()
    for j in range(p1 - 1, 1, -1):
        k = rf[:, j]
        d = (1 - k ** 2) ** (-1)
        rf[:, 1:j] = (rf[:, 1:j] - k[:, np.newaxis] * rf[:, j-1:0:-1]) * d[:, np.newaxis]

    return rf
