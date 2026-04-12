"""V_GAUSPROD - Calculate the product of Gaussians."""

from __future__ import annotations
import numpy as np


def v_gausprod(m, c=None) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculate the product of n d-dimensional multivariate Gaussians.

    Parameters
    ----------
    m : array_like
        Means, shape (d, n) - each column is the mean of one Gaussian.
    c : array_like, optional
        Covariance matrices. Can be:
        - (d, d, n) for full covariance
        - (d, n) for diagonal
        - (n,) for c*I
        - omitted for I

    Returns
    -------
    g : float
        Log gain (= log(integral)).
    u : ndarray
        Mean vector (d, 1).
    k : ndarray
        Covariance matrix (same form as input).
    """
    m = np.asarray(m, dtype=float)
    d, n = m.shape

    if c is None:
        c = np.ones(n)

    c = np.asarray(c, dtype=float)
    sc = c.shape

    if c.ndim < 3:
        if c.ndim == 1 and c.shape[0] == n and n != d:
            # Covariance matrices are multiples of the identity
            jj = 1
            jj2 = 2
            gj = np.zeros(n)
            while jj < n:
                for j_idx in range(jj, n, jj2):
                    k_idx = j_idx - jj
                    cjk = c[k_idx] + c[j_idx]
                    dm = m[:, k_idx] - m[:, j_idx]
                    gj[k_idx] = gj[k_idx] + gj[j_idx] - 0.5 * (d * np.log(cjk) + dm @ dm / cjk)
                    m[:, k_idx] = (c[k_idx] * m[:, j_idx] + c[j_idx] * m[:, k_idx]) / cjk
                    c[k_idx] = c[k_idx] * c[j_idx] / cjk
                jj = jj2
                jj2 = 2 * jj
            g = gj[0]
            k = c[0]
            u = m[:, 0]
        elif c.ndim == 2 and sc[0] == d and sc[1] == n:
            # Diagonal covariance matrices
            jj = 1
            jj2 = 2
            gj = np.zeros(n)
            while jj < n:
                for j_idx in range(jj, n, jj2):
                    k_idx = j_idx - jj
                    cjk = c[:, k_idx] + c[:, j_idx]
                    dm = m[:, k_idx] - m[:, j_idx]
                    ix = cjk > d * np.max(cjk) * np.finfo(float).eps
                    piv = np.zeros(d)
                    piv[ix] = 1.0 / cjk[ix]
                    gj[k_idx] = gj[k_idx] + gj[j_idx] - 0.5 * (np.log(np.prod(cjk)) + piv @ (dm ** 2))
                    m[:, k_idx] = piv * (c[:, k_idx] * m[:, j_idx] + c[:, j_idx] * m[:, k_idx])
                    c[:, k_idx] = c[:, k_idx] * piv * c[:, j_idx]
                jj = jj2
                jj2 = 2 * jj
            g = gj[0]
            k = c[:, 0]
            u = m[:, 0]
        else:
            # Handle c as scalar*I case when n==d (ambiguous) -- treat as multiples of I if 1D
            if c.ndim == 1 and c.shape[0] == n:
                jj = 1
                jj2 = 2
                gj = np.zeros(n)
                while jj < n:
                    for j_idx in range(jj, n, jj2):
                        k_idx = j_idx - jj
                        cjk = c[k_idx] + c[j_idx]
                        dm = m[:, k_idx] - m[:, j_idx]
                        gj[k_idx] = gj[k_idx] + gj[j_idx] - 0.5 * (d * np.log(cjk) + dm @ dm / cjk)
                        m[:, k_idx] = (c[k_idx] * m[:, j_idx] + c[j_idx] * m[:, k_idx]) / cjk
                        c[k_idx] = c[k_idx] * c[j_idx] / cjk
                    jj = jj2
                    jj2 = 2 * jj
                g = gj[0]
                k = c[0]
                u = m[:, 0]
            else:
                raise ValueError("Ambiguous covariance specification")
    else:
        # Full covariance matrices (d, d, n)
        jj = 1
        jj2 = 2
        gj = np.zeros(n)
        while jj < n:
            for j_idx in range(jj, n, jj2):
                k_idx = j_idx - jj
                cjk = c[:, :, k_idx] + c[:, :, j_idx]
                dm = m[:, k_idx] - m[:, j_idx]
                piv = np.linalg.pinv(cjk)
                gj[k_idx] = gj[k_idx] + gj[j_idx] - 0.5 * (np.log(np.linalg.det(cjk)) + dm @ piv @ dm)
                m[:, k_idx] = c[:, :, k_idx] @ piv @ m[:, j_idx] + c[:, :, j_idx] @ piv @ m[:, k_idx]
                c[:, :, k_idx] = c[:, :, k_idx] @ piv @ c[:, :, j_idx]
                c[:, :, k_idx] = 0.5 * (c[:, :, k_idx] + c[:, :, k_idx].T)
            jj = jj2
            jj2 = 2 * jj
        g = gj[0]
        k = c[:, :, 0]
        u = m[:, 0]

    g = g - 0.5 * (n - 1) * d * np.log(2 * np.pi)
    return g, u, k
