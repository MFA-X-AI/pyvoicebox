"""V_BESSELRATIO - Bessel function ratio I_{v+1}(x)/I_v(x)."""

from __future__ import annotations
import numpy as np


def v_besselratio(x, v=0, p=5) -> np.ndarray:
    """Calculate the Bessel function ratio besseli(v+1,x)/besseli(v,x).

    Parameters
    ----------
    x : array_like
        Bessel function argument.
    v : int, optional
        Denominator Bessel function order (default 0).
    p : int, optional
        Digits precision <=14 (default 5).

    Returns
    -------
    y : ndarray
        Value of the Bessel function ratio.
    """
    wm = [1, 1, 2, 4, 10, 20, 14, 21, 16, 22, 18, 23, 20, 25]
    nm = [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    p = min(max(int(np.ceil(p)), 1), 14)
    n = nm[p - 1]  # number of iterations
    u = max(v, wm[p - 1])  # minimum value of v for first stage

    x = np.asarray(x, dtype=float)
    s = x.shape
    x = x.ravel()

    r = np.zeros((x.size, n + 1))
    for i in range(1, n + 2):
        r[:, i - 1] = x / (u + i - 0.5 + np.sqrt((u + i + 0.5) ** 2 + x ** 2))

    for i in range(1, n + 1):
        for k in range(1, n - i + 2):  # k is k+1 in (20b)
            r[:, k - 1] = x / (
                u + k + np.sqrt((u + k) ** 2 + x ** 2 * r[:, k] / r[:, k - 1])
            )

    y = r[:, 0]
    for i in range(u, v, -1):
        y = 1.0 / (y + 2 * i / x)

    y[x == 0] = 0.0
    y[x == np.inf] = 1.0
    y = y.reshape(s)
    return y
