"""V_RANDFILT - Generate filtered Gaussian noise without initial transient."""

import numpy as np
from scipy.signal import lfilter
from scipy.linalg import toeplitz


def v_randfilt(pb, pa, ny=0, zi=None):
    """Generate filtered Gaussian noise without initial transient.

    Parameters
    ----------
    pb : array_like
        Numerator polynomial of discrete time filter.
    pa : array_like
        Denominator polynomial of discrete time filter.
    ny : int, optional
        Number of output samples required.
    zi : array_like, optional
        Initial filter state.

    Returns
    -------
    y : ndarray
        Filtered Gaussian noise.
    zf : ndarray
        Final filter state.
    u : ndarray
        State covariance factor (u@u' = state covariance).
    p : float
        Expected value of y(i)^2.
    """
    pb = np.asarray(pb, dtype=float).ravel()
    pa = np.asarray(pa, dtype=float).ravel()

    if pa[0] != 1:
        pb = pb / pa[0]
        pa = pa / pa[0]

    u = None
    p = None

    if zi is None or True:  # always compute for u/p outputs
        lb = len(pb)
        la = len(pa)
        k = max(la, lb) - 1  # filter order
        n = la - 1            # denominator order

        if k == 0:
            if zi is None:
                zi = np.array([])
            u = np.zeros((0, 0))
            p = pb[0] ** 2
        else:
            # Form controllability matrix
            q = np.zeros((k, k))
            _, q[:, 0] = lfilter(pb, pa, [1.0], zi=np.zeros(k))
            for i in range(1, k):
                _, q[:, i] = lfilter(pb, pa, [0.0], zi=q[:, i - 1])

            # Step-down procedure
            s = np.random.randn(k)
            ii = slice(k - n, k)
            if n > 0:
                m_mat = np.zeros((n, n))
                g = pa.copy()
                for i in range(n):
                    denom = np.sqrt((g[0] - g[-1]) * (g[0] + g[-1]))
                    if abs(denom) < 1e-15:
                        denom = 1e-15
                    g = (g[0] * g[:-1] - g[-1] * g[-1:0:-1]) / denom
                    m_mat[i, i:n] = g[:n - i]

                T = np.triu(toeplitz(pa[:n]))
                try:
                    s[ii] = T @ np.linalg.solve(m_mat, s[ii])
                except np.linalg.LinAlgError:
                    pass

                u = q.copy()
                try:
                    u[:, ii] = q[:, ii] @ T @ np.linalg.inv(m_mat)
                except np.linalg.LinAlgError:
                    pass
            else:
                u = q.copy()

            if zi is None:
                zi = q @ s

            p = u[0, :] @ u[0, :] + pb[0] ** 2

    if ny > 0:
        if len(zi) == 0:
            y, zf = lfilter(pb, pa, np.random.randn(ny)), zi
        else:
            y, zf = lfilter(pb, pa, np.random.randn(ny), zi=zi)
    else:
        zf = zi
        y = np.array([])

    return y, zf, u, p
