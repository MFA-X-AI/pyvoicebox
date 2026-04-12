"""V_LPCAR2LS - Convert AR polynomial to line spectrum pair frequencies."""

from __future__ import annotations
import numpy as np


def v_lpcar2ls(ar) -> np.ndarray:
    """Convert AR polynomial to line spectrum pair frequencies.

    Parameters
    ----------
    ar : array_like, shape (nf, p+1)
        Autoregressive coefficients.

    Returns
    -------
    ls : ndarray, shape (nf, p)
        Line spectrum pair frequencies in normalized Hz (0 to 0.5).
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    p = p1 - 1
    p2 = p // 2
    d = 0.5 / np.pi
    ls = np.zeros((nf, p))

    if p % 2:  # odd order
        for k in range(nf):
            aa = np.append(ar[k, :], 0.0)
            r = aa + aa[::-1]
            q = aa - aa[::-1]
            fr = np.sort(np.angle(np.roots(r)))
            # deconv q by [1, 0, -1]
            q_deconv = _deconv(q, np.array([1.0, 0.0, -1.0]))
            fq = np.sort(np.angle(np.roots(q_deconv)))
            fq = np.append(fq, 0.0)
            f = np.zeros(p + 1)
            f[0::2] = fr[p2 + 1:p + 1]
            f[1::2] = fq[p2:p]
            f = f[:p]
            ls[k, :] = d * f
    else:  # even order
        for k in range(nf):
            aa = np.append(ar[k, :], 0.0)
            r = aa + aa[::-1]
            q = aa - aa[::-1]
            r_deconv = _deconv(r, np.array([1.0, 1.0]))
            q_deconv = _deconv(q, np.array([1.0, -1.0]))
            fr = np.sort(np.angle(np.roots(r_deconv)))
            fq = np.sort(np.angle(np.roots(q_deconv)))
            f = np.zeros(p)
            f[0::2] = fr[p2:p]
            f[1::2] = fq[p2:p]
            ls[k, :] = d * f

    return ls


def _deconv(b, a):
    """Deconvolve polynomial a from b (equivalent to MATLAB deconv)."""
    b = np.array(b, dtype=float)
    a = np.array(a, dtype=float)
    na = len(a)
    nb = len(b)
    nq = nb - na + 1
    q = np.zeros(nq)
    r = b.copy()
    for i in range(nq):
        q[i] = r[i] / a[0]
        r[i:i + na] -= q[i] * a
    return q
