"""V_GAUSSMIXD - Marginal and conditional Gaussian mixture densities."""

import numpy as np
from .v_gaussmixp import v_gaussmixp


def v_gaussmixd(y, m, v, w, a=None, b=None, f=None, g=None, return_mixtures=False):
    """Compute conditional GMM densities.

    Parameters
    ----------
    y : array_like
        Conditioning data, shape (n, q).
    m : array_like
        Mixture means, shape (k, p).
    v : array_like
        Variances: diagonal (k, p) or full (p, p, k).
    w : array_like
        Mixture weights (k,).
    a : array_like, optional
        Conditioning transformation matrix.
    b : array_like, optional
        Dimension selection indices (1-based, MATLAB convention).
    f : array_like, optional
        Output transformation matrix.
    g : array_like, optional
        Output offset or dimension selection.
    return_mixtures : bool, optional
        If True, return per-mixture means and weights.

    Returns
    -------
    mz : ndarray
        Global mean of z for each y (n, r), or per-mixture if return_mixtures=True.
    vz : ndarray
        Global or per-mixture covariances.
    wz : ndarray
        Mixture weights (only if return_mixtures=True).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    n, q = y.shape

    m = np.asarray(m, dtype=float)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    k, p = m.shape

    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float).ravel()

    fv = v.ndim > 2 or (v.ndim == 2 and v.shape[0] > k)

    # Set up conditioning transformation
    anull = a is None
    if anull:
        a_mat = np.eye(p)
        if b is None:
            b_idx = np.arange(q)
        else:
            b_idx = np.asarray(b, dtype=int).ravel() - 1  # Convert to 0-based
        a_mat = a_mat[b_idx, :]
        b_saved = b_idx.copy()
        b_vec = np.zeros(q)
    else:
        a_mat = np.asarray(a, dtype=float)
        if b is None:
            b_vec = np.zeros(q)
        else:
            b_vec = np.asarray(b, dtype=float).ravel()

    # Set up output transformation
    if f is None:
        f_mat = np.eye(p)
        if g is None:
            if anull:
                # Complement of selected dimensions
                all_dims = set(range(p))
                f_mat = np.eye(p)
                remaining = sorted(all_dims - set(b_saved))
                f_mat = f_mat[remaining, :]
            else:
                f_mat = f_mat[q:, :]
        else:
            g_idx = np.asarray(g, dtype=int).ravel() - 1  # Convert to 0-based
            f_mat = f_mat[g_idx, :]
        r = f_mat.shape[0]
        g_vec = np.zeros(r)
    else:
        f_mat = np.asarray(f, dtype=float)
        r = f_mat.shape[0]
        if g is None:
            g_vec = np.zeros(r)
        else:
            g_vec = np.asarray(g, dtype=float).ravel()

    yb = y - b_vec[np.newaxis, :]

    # Find mixture weights given y
    lp, wz_all = v_gaussmixp(yb, m, v, w, a_mat)[:2]

    mz = np.zeros((n, r, k))

    for i in range(k):
        if fv:
            vi = v[:, :, i]
        else:
            vi = np.diag(v[i, :])

        avat = a_mat @ vi @ a_mat.T
        hi = vi @ a_mat.T @ np.linalg.solve(avat, np.eye(avat.shape[0]))
        vzi = f_mat @ (vi - hi @ a_mat @ vi) @ f_mat.T
        mi = m[i, :]
        m0 = (mi - mi @ a_mat.T @ hi.T) @ f_mat.T + g_vec
        mz[:, :, i] = m0[np.newaxis, :] + yb @ hi.T @ f_mat.T

    if not return_mixtures:
        # Compute global mean
        mt = np.zeros((n, r))
        for i in range(k):
            mt += mz[:, :, i] * wz_all[:, i:i + 1]

        # Compute global variance
        vz_global = np.zeros((r, r, n))
        for idx_n in range(n):
            for i in range(k):
                if fv:
                    vi_full = v[:, :, i]
                else:
                    vi_full = np.diag(v[i, :])
                avat = a_mat @ vi_full @ a_mat.T
                hi = vi_full @ a_mat.T @ np.linalg.solve(avat, np.eye(avat.shape[0]))
                vzi = f_mat @ (vi_full - hi @ a_mat @ vi_full) @ f_mat.T
                dm = mz[idx_n, :, i] - mt[idx_n, :]
                vz_global[:, :, idx_n] += wz_all[idx_n, i] * (vzi + np.outer(dm, dm))

        return mt, vz_global
    else:
        # Return per-mixture results
        vz_mix = np.zeros((r, r, k))
        for i in range(k):
            if fv:
                vi = v[:, :, i]
            else:
                vi = np.diag(v[i, :])
            avat = a_mat @ vi @ a_mat.T
            hi = vi @ a_mat.T @ np.linalg.solve(avat, np.eye(avat.shape[0]))
            vz_mix[:, :, i] = f_mat @ (vi - hi @ a_mat @ vi) @ f_mat.T

        mz_out = np.transpose(mz, (2, 1, 0))  # (k, r, n)
        return mz_out, vz_mix, wz_all
