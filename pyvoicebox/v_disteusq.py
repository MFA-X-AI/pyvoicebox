"""V_DISTEUSQ - Squared Euclidean distance matrix."""

from __future__ import annotations
import numpy as np


def v_disteusq(x, y, mode='', w=None) -> np.ndarray:
    """Calculate squared Euclidean or Mahalanobis distance.

    Parameters
    ----------
    x : array_like
        First set of vectors, shape (nx, p).
    y : array_like
        Second set of vectors, shape (ny, p).
    mode : str, optional
        'x' = full distance matrix; 'd' = pairwise; 's' = take sqrt.
    w : array_like, optional
        Weighting matrix or vector.

    Returns
    -------
    d : ndarray
        Distance matrix or vector.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    nx, p = x.shape
    ny = y.shape[0]

    if 'd' in mode or ('x' not in mode and nx == ny):
        # Pairwise distance
        nn = min(nx, ny)
        z = x[:nn, :] - y[:nn, :]
        if w is None:
            d = np.sum(z * z, axis=1)
        else:
            w = np.asarray(w, dtype=float)
            if w.ndim == 1:
                d = np.sum(z * w[np.newaxis, :] * z, axis=1)
            else:
                d = np.sum((z @ w) * z, axis=1)
    else:
        # Full distance matrix (nx, ny)
        if w is None:
            # Efficient computation: ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x*y'
            xx = np.sum(x * x, axis=1, keepdims=True)
            yy = np.sum(y * y, axis=1, keepdims=True)
            d = xx + yy.T - 2.0 * (x @ y.T)
            d = np.maximum(d, 0.0)  # avoid tiny negatives from rounding
        else:
            w = np.asarray(w, dtype=float)
            if w.ndim == 1:
                xw = x * w[np.newaxis, :]
                xx = np.sum(xw * x, axis=1, keepdims=True)
                yy = np.sum(y * w[np.newaxis, :] * y, axis=1, keepdims=True)
                d = xx + yy.T - 2.0 * (xw @ y.T)
            else:
                xw = x @ w
                xx = np.sum(xw * x, axis=1, keepdims=True)
                yw = y @ w
                yy = np.sum(yw * y, axis=1, keepdims=True)
                d = xx + yy.T - 2.0 * (xw @ y.T)
            d = np.maximum(d, 0.0)

    if 's' in mode:
        d = np.sqrt(d)

    return d
