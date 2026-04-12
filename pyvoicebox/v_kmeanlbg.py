"""V_KMEANLBG - K-means using Linde-Buzo-Gray algorithm."""

from __future__ import annotations
import numpy as np
from .v_kmeans import v_kmeans


def v_kmeanlbg(d, k) -> tuple[np.ndarray, float, np.ndarray]:
    """Vector quantization using the Linde-Buzo-Gray algorithm.

    Parameters
    ----------
    d : array_like
        Data vectors (one per row), shape (n, p).
    k : int
        Number of centres required.

    Returns
    -------
    x : ndarray
        Output centres, shape (k, p).
    esq : float
        Mean squared error.
    j : ndarray
        Cluster assignments (0-based).
    """
    d = np.asarray(d, dtype=float)
    nc = d.shape[1]

    x, esq, j, _ = v_kmeans(d, 1)
    m_count = 1
    while m_count < k:
        n_split = min(m_count, k - m_count)
        m_count = m_count + n_split
        e = 1e-4 * np.sqrt(esq) * np.random.rand(1, nc)
        x0 = np.vstack([
            x[:n_split, :] + np.tile(e, (n_split, 1)),
            x[:n_split, :] - np.tile(e, (n_split, 1)),
            x[n_split:m_count - n_split, :]
        ])
        x, esq, j, _ = v_kmeans(d, m_count, x0)

    return x, esq, j
