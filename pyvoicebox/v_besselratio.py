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

    # Identify edge cases upfront
    edge_zero = x == 0
    edge_inf = x == np.inf
    finite = ~edge_zero & ~edge_inf

    # Work only on finite, nonzero values
    xf = x[finite]
    rf = np.zeros((xf.size, n + 1))
    for i in range(1, n + 2):
        rf[:, i - 1] = xf / (u + i - 0.5 + np.sqrt((u + i + 0.5) ** 2 + xf ** 2))

    for i in range(1, n + 1):
        for k in range(1, n - i + 2):
            rf[:, k - 1] = xf / (
                u + k + np.sqrt((u + k) ** 2 + xf ** 2 * rf[:, k] / rf[:, k - 1])
            )

    yf = rf[:, 0]
    for i in range(u, v, -1):
        yf = 1.0 / (yf + 2 * i / xf)

    # Assemble result
    y = np.empty_like(x)
    y[finite] = yf
    y[edge_zero] = 0.0
    y[edge_inf] = 1.0
    y = y.reshape(s)
    return y
