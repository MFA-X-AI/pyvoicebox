"""V_RANDVEC - Generate random vectors from a GMM distribution."""

from __future__ import annotations
import numpy as np
from .v_randiscr import v_randiscr


def v_randvec(n, m, c=None, w=None, mode='g') -> tuple[np.ndarray, np.ndarray]:
    """Generate real or complex GMM/lognormal random vectors.

    Parameters
    ----------
    n : int
        Number of points to generate.
    m : array_like
        Mixture means, shape (k, p).
    c : array_like, optional
        Covariances: diagonal (k, p), full (p, p, k), or scalar per mixture.
    w : array_like, optional
        Mixture weights (k,). Default: equal weights.
    mode : str, optional
        'g' = real Gaussian (default), 'c' = complex Gaussian, 'l' = lognormal.

    Returns
    -------
    x : ndarray
        Output data, shape (n, p).
    kx : ndarray
        Mixture number for each row (0-based), shape (n,).
    """
    m = np.asarray(m, dtype=float)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    sm = m.shape
    k = sm[0]
    p = sm[1]

    if c is None:
        c = np.ones_like(m)
    c = np.asarray(c, dtype=float)
    sc = c.shape

    fullc = c.ndim > 2 or (c.ndim == 2 and sc[0] > k)

    if w is None:
        w = np.ones(k)
    else:
        w = np.asarray(w, dtype=float).ravel()

    if isinstance(mode, str) and len(mode) > 0:
        ty = mode[0]
    else:
        ty = 'g'

    x = np.zeros((n, p))

    if k > 1:
        kx = v_randiscr(w, n)
    else:
        kx = np.zeros(n, dtype=int)

    for kk_idx in range(k):
        nx = np.where(kx == kk_idx)[0]
        nn = len(nx)
        if nn == 0:
            continue

        mm = m[kk_idx, :]
        if fullc:
            cc = c[:, :, kk_idx].copy()
            if ty == 'l':
                cc = np.log(1 + cc / np.outer(mm, mm))
                mm = np.log(mm) - 0.5 * np.diag(cc)
        else:
            cc = c[kk_idx, :].copy()
            if ty == 'l':
                cc = np.log(1 + cc / mm ** 2)
                mm = np.log(mm) - 0.5 * cc

        if ty == 'c':
            q = np.sqrt(0.5)
            xx = q * np.random.randn(nn, p) + 1j * q * np.random.randn(nn, p)
        else:
            xx = np.random.randn(nn, p)

        if fullc:
            cc_sym = (cc + cc.T) / 2
            eigvals, eigvecs = np.linalg.eigh(cc_sym)
            xx = xx * np.sqrt(np.abs(eigvals))[np.newaxis, :] @ eigvecs.T + mm[np.newaxis, :]
        else:
            xx = xx * np.sqrt(np.abs(cc))[np.newaxis, :] + mm[np.newaxis, :]

        x[nx, :] = xx

    if ty == 'l':
        x = np.exp(x)

    return x, kx
