"""V_MAXGAUSS - Gaussian approximation to the max of a Gaussian vector."""

from __future__ import annotations
import numpy as np
from scipy.special import erfc


def v_maxgauss(m, c=None, d=None) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Determine Gaussian approximation to max of a Gaussian vector.

    Parameters
    ----------
    m : array_like
        Mean vector of length N.
    c : array_like, optional
        Covariance matrix (N,N) or vector of variances (N,).
    d : array_like, optional
        Covariance w.r.t. some other variables (N,K).

    Returns
    -------
    u : float
        Mean of max(x).
    v : float
        Variance of max(x).
    p : ndarray
        Probability of each element being the max (N,).
    r : ndarray
        Covariance between max(x) and the d-variables (1,K).
    """
    nrh = -np.sqrt(0.5)
    kpd = np.sqrt(0.5 / np.pi)

    m = np.asarray(m, dtype=float).ravel()
    nm = len(m)

    if c is None:
        c = np.eye(nm)
    else:
        c = np.asarray(c, dtype=float)
        if c.ndim == 1 or (c.ndim == 2 and c.shape[1] == 1 and c.shape[0] == nm):
            c = np.diag(c.ravel())
        elif c.ndim == 0:
            c = np.diag(np.full(nm, float(c)))

    p = np.eye(nm)

    # Remove negative infinities
    ix = np.where(m == -np.inf)[0]
    if len(ix) > 0:
        m = np.delete(m, ix)
        c = np.delete(np.delete(c, ix, axis=0), ix, axis=1)
        p = np.delete(p, ix, axis=1)
        nm = len(m)

    while nm > 1:
        # Build strict upper triangle indices
        ix_list = []
        for col in range(nm):
            for row in range(col):
                ix_list.append((row, col))

        # Calculate scaled differences in means
        cd = np.diag(c)
        gm_arr = []
        gv_arr = []
        for row, col in ix_list:
            gm_arr.append(m[row] - m[col])
            gv_arr.append(cd[row] + cd[col] - c[row, col] - c[col, row])

        gm_arr = np.array(gm_arr)
        gv_arr = np.array(gv_arr)

        jx_zero = np.where(gv_arr <= 0)[0]
        if len(jx_zero) > 0:
            jx = jx_zero[0]
            i, j = ix_list[jx]
            dm = gm_arr[jx]
            if dm > 0:
                m = np.delete(m, j)
                c = np.delete(np.delete(c, j, axis=0), j, axis=1)
                p = np.delete(p, j, axis=1)
            else:
                m = np.delete(m, i)
                c = np.delete(np.delete(c, i, axis=0), i, axis=1)
                p = np.delete(p, i, axis=1)
        else:
            ratios = gm_arr ** 2 / gv_arr
            jx = np.argmax(ratios)
            i, j = ix_list[jx]

            dm = gm_arr[jx]
            ds = np.sqrt(gv_arr[jx])
            dms = dm / ds
            q = 0.5 * erfc(nrh * dms)
            f = kpd * np.exp(-0.5 * dms ** 2)

            mi_val = m[i]
            mj_val = m[j]
            u_val = dm * q + mj_val + ds * f
            v_val = (mi_val + mj_val - u_val) * u_val + cd[i] * q + cd[j] * (1 - q) - mi_val * mj_val

            m[i] = u_val
            c[i, :] = q * c[i, :] + (1 - q) * c[j, :]
            c[:, i] = c[i, :]
            c[i, i] = v_val
            p[:, i] = q * p[:, i] + (1 - q) * p[:, j]

            m = np.delete(m, j)
            c = np.delete(np.delete(c, j, axis=0), j, axis=1)
            p = np.delete(p, j, axis=1)

        nm = len(m)

    u = m[0]
    v = c[0, 0]
    p = p.ravel() / np.sum(p)

    r = None
    if d is not None:
        d = np.asarray(d, dtype=float)
        r = p @ d
        r = r.reshape(1, -1)

    return u, v, p, r
