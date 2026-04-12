"""V_GAUSSMIXP - Calculate log probability densities from a Gaussian mixture model."""

from __future__ import annotations
import numpy as np


def v_gaussmixp(y, m, v=None, w=None, a=None, b=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate log probability densities from a Gaussian mixture model.

    Parameters
    ----------
    y : array_like
        Input data, shape (n, q).
    m : array_like
        Mixture means, shape (k, p).
    v : array_like, optional
        Variances: diagonal (k, p) or full (p, p, k). Default: identity.
    w : array_like, optional
        Mixture weights (k,), must sum to 1. Default: uniform.
    a : array_like, optional
        Transformation matrix (q, p). y = x*A' + B'.
    b : array_like, optional
        Offset vector (q,).

    Returns
    -------
    lp : ndarray
        Log probability of each data point (n,).
    rp : ndarray
        Relative probability of each mixture (n, k).
    kh : ndarray
        Index of highest probability mixture (n,).
    kp : ndarray
        Relative probability of highest mixture (n,).
    """
    m = np.asarray(m, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if m.ndim == 1:
        m = m.reshape(1, -1)

    k, p = m.shape
    n, q = y.shape

    if w is None:
        w = np.full(k, 1.0 / k)
    else:
        w = np.asarray(w, dtype=float).ravel()

    if v is None:
        v = np.ones((k, p))
    else:
        v = np.asarray(v, dtype=float)

    # Determine if full covariance
    fv = v.ndim > 2 or (v.ndim == 2 and v.shape[0] > k)

    # Apply transformations
    if a is not None:
        a = np.asarray(a, dtype=float)
        if b is not None:
            b = np.asarray(b, dtype=float).ravel()
            m = m @ a.T + b[np.newaxis, :]
        else:
            m = m @ a.T
        if fv:
            v_new = np.zeros((q, q, k))
            for ik in range(k):
                v_new[:, :, ik] = a @ v[:, :, ik] @ a.T
            v = v_new
        else:
            # Check if a preserves diagonality
            if np.all(np.sum(a != 0, axis=1) == 1):
                v_new = np.zeros((k, q))
                for ik in range(k):
                    v_new[ik, :] = v[ik, :] @ (a ** 2).T
                v = v_new
            else:
                v_new = np.zeros((q, q, k))
                for ik in range(k):
                    v_new[:, :, ik] = (a * v[ik, :][np.newaxis, :]) @ a.T
                v = v_new
                fv = True
        q_dim = q
    elif q < p or (a is None and b is not None):
        if b is None:
            b_idx = np.arange(q)
        else:
            b_idx = np.asarray(b, dtype=int).ravel()
            if np.issubdtype(type(b_idx[0]), np.integer) or np.all(b_idx == b_idx.astype(int)):
                b_idx = b_idx.astype(int) - 1 if np.min(b_idx) >= 1 else b_idx.astype(int)
        m = m[:, b_idx]
        if fv:
            v = v[np.ix_(b_idx, b_idx, np.arange(k))]
        else:
            v = v[:, b_idx]
        q_dim = q
    else:
        q_dim = q

    fv = v.ndim > 2 or (v.ndim == 2 and v.shape[0] > k)

    lp = np.zeros(n)
    rp = np.zeros((n, k))

    if n > 0:
        if not fv:
            # Diagonal covariance matrices
            vi = -0.5 / v  # (k, q)
            lvm = np.log(w) - 0.5 * np.sum(np.log(v), axis=1)  # (k,)

            # Compute log probabilities for all data points and all mixtures
            # py(k, n) = sum((y - m_k)^2 * vi_k) + lvm_k
            py = np.zeros((k, n))
            for ik in range(k):
                diff = y - m[ik, :]  # (n, q)
                py[ik, :] = np.sum(diff ** 2 * vi[ik, :], axis=1) + lvm[ik]

            mx = np.max(py, axis=0)  # (n,)
            px = np.exp(py - mx[np.newaxis, :])  # (k, n)
            ps = np.sum(px, axis=0)  # (n,)
            rp = (px / ps[np.newaxis, :]).T  # (n, k)
            lp = np.log(ps) + mx
        else:
            # Full covariance matrices
            vi = np.zeros((q_dim * k, q_dim))
            vim = np.zeros(q_dim * k)
            mtk = np.zeros(q_dim * k)
            lvm = np.zeros(k)

            for ik in range(k):
                vk = v[:, :, ik]
                eigvals, eigvecs = np.linalg.eigh(vk)
                if np.any(eigvals <= 0):
                    raise ValueError(f'Covariance matrix for mixture {ik} is not positive definite')
                vik = -0.5 * eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                sl = slice(ik * q_dim, (ik + 1) * q_dim)
                vi[sl, :] = vik
                vim[sl] = vik @ m[ik, :]
                mtk[sl] = m[ik, :]
                lvm[ik] = np.log(w[ik]) - 0.5 * np.sum(np.log(eigvals))

            # Compute for all points
            py = np.zeros((k, n))
            for ik in range(k):
                sl = slice(ik * q_dim, (ik + 1) * q_dim)
                vik = vi[sl, :]
                vim_k = vim[sl]
                mk = m[ik, :]
                for i in range(n):
                    yi = y[i, :]
                    py[ik, i] = (vik @ yi - vim_k) @ (yi - mk) + lvm[ik]

            mx = np.max(py, axis=0)
            px = np.exp(py - mx[np.newaxis, :])
            ps = np.sum(px, axis=0)
            rp = (px / ps[np.newaxis, :]).T
            lp = np.log(ps) + mx

        lp = lp - 0.5 * q_dim * np.log(2 * np.pi)

    kh = np.argmax(rp, axis=1)
    kp = rp[np.arange(n), kh]

    return lp, rp, kh, kp
