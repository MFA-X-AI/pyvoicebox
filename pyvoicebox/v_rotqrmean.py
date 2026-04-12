"""V_ROTQRMEAN - calculate mean rotation of quaternion array."""

from __future__ import annotations
import numpy as np


def v_rotqrmean(q) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate the mean rotation of a quaternion array.

    Parameters
    ----------
    q : array_like, shape (4, n)
        Normalized real quaternion array.

    Returns
    -------
    y : ndarray, shape (4,)
        Normalized mean quaternion.
    s : ndarray, shape (n,)
        Sign vector such that y = normalize(q @ s).
    v : float
        Average squared deviation from the mean quaternion.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q.reshape(4, 1)

    mmax = 10  # number of n-best hypotheses to keep
    nq = q.shape[1]

    mkx = np.zeros((nq, mmax), dtype=int)
    msum = np.zeros((4, 2 * mmax))
    msum[:, 0] = q[:, 0]

    ix = np.arange(mmax)
    jx = np.arange(mmax, 2 * mmax)

    for i in range(1, nq):
        qi = q[:, i:i + 1]  # (4, 1)
        msum[:, jx] = msum[:, ix] - qi
        msum[:, ix] = msum[:, ix] + qi

        # Sort by squared norm (descending)
        norms = np.sum(msum ** 2, axis=0)
        kx = np.argsort(-norms)

        mkx[i, :] = kx[:mmax]
        msum[:, ix] = msum[:, kx[:mmax]]

    y = msum[:, 0]
    y = y / np.sqrt(y @ y)

    # Traceback to find signs
    s = np.zeros(nq)
    k = 0
    for i in range(nq - 1, 0, -1):
        neg = int(mkx[i, k] >= mmax)
        s[i] = neg
        k = mkx[i, k] - mmax * neg

    s = 1.0 - 2.0 * s

    v = float(np.sum(np.mean((q - np.outer(y, s)) ** 2, axis=1)))

    return y, s, v
