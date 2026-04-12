"""V_KMEANHAR - K-harmonic means clustering algorithm."""

from __future__ import annotations
import numpy as np
from .v_rnsubset import v_rnsubset


def v_kmeanhar(d, k, l=None, e=None, x0='f') -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Vector quantization using K-harmonic means algorithm.

    Parameters
    ----------
    d : array_like
        Data vectors, shape (n, p).
    k : int
        Number of centres required.
    l : float, optional
        Max iterations (integer part) + stopping threshold (fractional part).
        Default: 50.001.
    e : int, optional
        Exponent in the cost function (default 4).
    x0 : str or array_like, optional
        Initial centres or initialization method ('f' or 'p').

    Returns
    -------
    x : ndarray
        Output centres, shape (k, p).
    g : float
        Final performance criterion (normalized by n).
    xn : ndarray
        Nearest centre for each input point (0-based).
    gg : ndarray
        Performance criterion at each iteration.
    """
    d = np.asarray(d, dtype=float)
    n, p = d.shape

    if e is None:
        e = 4
    if l is None:
        l = 50 + 1e-3

    sd = 5  # number of times we must be below threshold

    # Initialize
    if isinstance(x0, str):
        if k < n:
            if 'p' in x0:
                ix = np.random.randint(0, k, size=n)
                forced = v_rnsubset(k, n)
                ix[forced] = np.arange(k)
                x = np.zeros((k, p))
                for i in range(k):
                    mask = ix == i
                    if np.any(mask):
                        x[i, :] = np.mean(d[mask, :], axis=0)
            else:
                x = d[v_rnsubset(k, n), :]
        else:
            x = d[np.arange(k) % n, :]
    else:
        x = np.asarray(x0, dtype=float).copy()

    eh = e / 2.0
    th = l - np.floor(l)
    max_iter = int(np.floor(l)) + 1  # extra loop to compute final value
    if max_iter <= 1:
        max_iter = 100
    if th == 0:
        th = -1

    gg = np.zeros(max_iter + 1)
    xn = np.zeros(n, dtype=int)

    ss = sd + 1
    g = 0

    for j in range(max_iter):
        g1 = g
        x1 = x.copy()

        # Compute squared distances (n x k)
        # py(k, n) in MATLAB, here we compute as (n, k) then transpose
        diff = d[:, np.newaxis, :] - x[np.newaxis, :, :]  # (n, k, p)
        py = np.sum(diff ** 2, axis=2).T  # (k, n)

        dm = np.min(py, axis=0)  # (n,)
        xn = np.argmin(py, axis=0)  # (n,)

        dmk = np.tile(dm, (k, 1))  # (k, n)
        dq = py > dmk
        pr = np.ones((k, n))
        pr[dq] = dmk[dq] / py[dq]

        pe = pr ** eh
        se = np.sum(pe, axis=0)  # (n,)

        xf = dm ** (eh - 1) / se  # (n,)
        g = np.dot(xf, dm)  # scalar

        xg = xf / se  # (n,)
        qik = xg[np.newaxis, :] * pe * pr  # (k, n)
        qk = np.sum(qik, axis=1)  # (k,)
        xs = qik @ d  # (k, p)

        gg[j] = g
        x = xs / qk[:, np.newaxis]

        if g1 - g <= th * g1:
            ss -= 1
            if ss <= 0:
                break
        else:
            ss = sd

    j_final = j + 1
    gg = gg[:j_final] * k / n
    g_final = gg[-1]

    # Go back to previous x values
    x = x1

    return x, g_final, xn, gg
