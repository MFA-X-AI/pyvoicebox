"""V_GAUSSMIXM - Estimate mean and variance of the magnitude of a GMM."""

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm


def v_gaussmixm(m, v=None, w=None, z=None):
    """Estimate mean and variance of the magnitude of a GMM.

    Parameters
    ----------
    m : array_like
        Mixture means, shape (k, p).
    v : array_like, optional
        Variances: diagonal (k, p) or full (p, p, k).
    w : array_like, optional
        Mixture weights (k,).
    z : array_like, optional
        Origins, shape (t, p).

    Returns
    -------
    mm : ndarray
        Mean of |x-z|, shape (t,).
    mc : ndarray
        Variance (or covariance) of |x-z|.
    """
    m = np.asarray(m, dtype=float)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    k, p = m.shape

    if z is None:
        z = np.zeros((1, p))
    else:
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            z = z.reshape(1, -1)

    if w is None:
        w = np.ones(k)
    else:
        w = np.asarray(w, dtype=float).ravel()

    if v is None:
        v = np.ones((k, p))
    else:
        v = np.asarray(v, dtype=float)

    t = z.shape[0]
    w = w / np.sum(w)

    if p == 1:
        # Exact 1D formula
        s = np.sqrt(v.ravel())
        mt = m[:, np.newaxis] - z.T  # (k, t)
        mts = mt / s[:, np.newaxis]
        ncdf = norm.cdf(-mts)
        npdf = norm.pdf(-mts)
        mm = ((mts * (1 - 2 * ncdf) + 2 * npdf).T @ (s * w)).ravel()

        mc = np.diag(((mts ** 2 + 1).T @ (v.ravel() * w)).ravel())
        if t > 1:
            for it in range(t):
                for jt in range(it):
                    mc[it, jt] = w @ (
                        (v.ravel() + mt[:, it] * mt[:, jt]) * (1 - 2 * np.abs(ncdf[:, it] - ncdf[:, jt]))
                        + 2 * s * np.sign(mt[:, jt] - mt[:, it]) * (npdf[:, it] * mt[:, jt] - npdf[:, jt] * mt[:, it])
                    )
                    mc[jt, it] = mc[it, jt]
        mc = mc - np.outer(mm, mm)
    else:
        fullv = v.ndim > 2 or (v.ndim == 2 and v.shape[0] > k)

        if fullv:
            diag_idx = np.arange(p)
            ms = np.zeros((k, t))
            for i in range(k):
                ms[i, :] = np.sum(m[i, :] ** 2) + np.sum(np.diag(v[:, :, i])) - 2 * m[i, :] @ z.T + np.sum(z ** 2, axis=1)
            vs = np.zeros((k, t))
            for i in range(k):
                si = v[:, :, i]
                vsc_i = 2 * np.trace(si @ si)
                for jt in range(t):
                    zmi = m[i, :] - z[jt, :]
                    vs[i, jt] = vsc_i + 4 * zmi @ si @ zmi
        else:
            ms = (np.sum(m ** 2, axis=1) + np.sum(v, axis=1))[:, np.newaxis] - 2 * m @ z.T + np.sum(z ** 2, axis=1)[np.newaxis, :]
            vsc = np.sum(v * (2 * v + 4 * m ** 2), axis=1)
            vmz = 4 * (v * m) @ z.T
            vs = vsc[:, np.newaxis] - 2 * vmz + 4 * v @ (z ** 2).T

        nm = ms ** 2 / vs
        mmk = np.exp(gammaln(nm + 0.5) - gammaln(nm)) * np.sqrt(ms / nm)
        mm = (mmk.T @ w).ravel()
        mc = (ms.T @ w - mm ** 2).ravel()

    return mm, mc
