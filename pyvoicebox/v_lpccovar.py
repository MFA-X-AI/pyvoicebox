"""V_LPCCOVAR - Perform covariance LPC analysis."""

import numpy as np


def v_lpccovar(s, p=12, t=None):
    """Perform covariance LPC analysis.

    Parameters
    ----------
    s : array_like, shape (ns,)
        Input signal.
    p : int, optional
        LPC order. Default is 12.
    t : array_like, optional
        Frame specification. Default is [p+1, len(s)] (1-based).

    Returns
    -------
    ar : ndarray, shape (nf, p+1)
        AR coefficients with ar[:, 0] = 1.
    e : ndarray, shape (nf, 4)
        [Er, Es, Pr, Ps]: energy and power in residual and signal.
    dc : ndarray, shape (nf,)
        DC component (only meaningful if computed).
    """
    s = np.asarray(s, dtype=float).ravel()
    ns = len(s)

    if t is None:
        t = np.array([[p + 1, ns]])  # 1-based
    else:
        t = np.atleast_2d(np.asarray(t, dtype=float))

    nf, ng = t.shape
    if ng % 2 == 1:
        # Add end column
        ends = np.concatenate([t[1:, 0] - 1, [ns]])
        t = np.column_stack([t, ends])

    ar = np.zeros((nf, p + 1))
    ar[:, 0] = 1.0
    e = np.zeros((nf, 4))
    dc = np.zeros(nf)

    for jf in range(nf):
        tj = t[jf, :]

        # Build sample indices (1-based to 0-based)
        ta = int(np.ceil(tj[0]))
        tb = int(np.floor(tj[1]))
        cs = np.arange(ta, tb + 1)  # 1-based indices

        for js in range(2, len(tj), 2):
            ta = int(np.ceil(tj[js]))
            tb = int(np.floor(tj[js + 1]))
            cs = np.concatenate([cs, np.arange(ta, tb + 1)])

        nc = len(cs)
        pp = min(p, nc)

        # Build data matrix (1-based indices: s(cs - 1), s(cs - 2), ..., s(cs - p))
        dm = np.zeros((nc, pp))
        for col in range(pp):
            dm[:, col] = s[cs - 1 - (col + 1)]  # cs is 1-based, so cs-1 is 0-based index

        sc = s[cs - 1]  # 0-based indexing
        aa = np.linalg.lstsq(dm, sc, rcond=None)[0]
        ar[jf, 1:pp + 1] = -aa

        residual = sc - dm @ aa
        e[jf, 0] = np.real(np.dot(sc, residual))
        e[jf, 1] = np.real(np.dot(sc, sc))
        e[jf, 2:4] = e[jf, :2] / nc

    return ar, e, dc
