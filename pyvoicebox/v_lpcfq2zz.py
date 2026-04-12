"""V_LPCFQ2ZZ - Convert frequencies and Q factors to z-plane poles."""

from __future__ import annotations
import numpy as np


def v_lpcfq2zz(f, q=None) -> np.ndarray:
    """Convert frequencies and Q factors to z-plane poles.

    Parameters
    ----------
    f : array_like, shape (nf, pf)
        Frequencies in normalized Hz.
    q : array_like, shape (nf, pq), optional
        Q factors.

    Returns
    -------
    zz : ndarray, shape (nf, pf+pq)
        Z-plane poles.
    """
    f = np.atleast_2d(np.asarray(f, dtype=float))
    nf, pf = f.shape

    if q is None:
        pq = 0
    else:
        q = np.atleast_2d(np.asarray(q, dtype=float))
        pq = q.shape[1]

    zz = np.zeros((nf, pf + pq), dtype=complex)

    if pq > 0:
        ii = np.arange(pq)
        zz[:, 2 * ii] = np.exp(np.pi * f[:, :pq] * (2j - q ** (-1)))
        zz[:, 2 * ii + 1] = np.conj(zz[:, 2 * ii])

    if pf > pq:
        ii = np.arange(pq, pf)
        zz[:, ii + pq] = np.exp(-2 * np.pi * f[:, ii])

    return zz
