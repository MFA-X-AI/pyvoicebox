"""V_GAUSSMIXG - Global mean, variance and mode of a GMM (computation only)."""

import numpy as np
from .v_gaussmixp import v_gaussmixp


def v_gaussmixg(m, v=None, w=None, n_modes=1):
    """Compute global mean, variance and modes of a GMM.

    Parameters
    ----------
    m : array_like
        Mixture means, shape (k, p).
    v : array_like, optional
        Variances: diagonal (k, p) or full (p, p, k). Default: ones.
    w : array_like, optional
        Mixture weights. Default: uniform.
    n_modes : int, optional
        Maximum number of modes to find (default 1).

    Returns
    -------
    mg : ndarray
        Global mean, shape (p,).
    vg : ndarray
        Global covariance, shape (p, p).
    pg : ndarray
        Sorted list of modes, shape (n_modes, p).
    pv : ndarray
        Log PDF at the modes, shape (n_modes,).
    """
    m = np.asarray(m, dtype=float)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    k, p = m.shape

    if v is None:
        v = np.ones((k, p))
    else:
        v = np.asarray(v, dtype=float)

    if w is None:
        w = np.ones(k)
    else:
        w = np.asarray(w, dtype=float).ravel()

    full = v.ndim > 2 or (k == 1 and v.size > p)
    if full and p == 1:
        v = v.reshape(k, 1)
        full = False

    w = w / np.sum(w)

    # Global mean
    mg = w @ m

    # Global covariance
    mz = m - mg[np.newaxis, :]
    if full:
        vg = mz.T @ (mz * w[:, np.newaxis]) + np.reshape(
            np.reshape(v, (p * p, k)) @ w, (p, p)
        )
    else:
        vg = mz.T @ (mz * w[:, np.newaxis]) + np.diag(w @ v)

    # Mode finding using fixed-point iteration
    nfp = 2
    maxloop = 60
    ssf = 0.1

    pg = m.copy()  # initialize mode candidates to mixture means
    nx = k

    if not full:
        vi = -0.5 / v
        lvm = np.log(w) - 0.5 * np.sum(np.log(v), axis=1)
        vim = vi * m
        ss_sep = np.sqrt(np.min(v, axis=0)) * ssf / np.sqrt(p)
    else:
        vi_stack = np.zeros((p * k, p))
        vim_stack = np.zeros(p * k)
        mtk_stack = np.zeros(p * k)
        lvm = np.zeros(k)
        for i in range(k):
            eigvals, eigvecs = np.linalg.eigh(v[:, :, i])
            if np.any(eigvals <= 0):
                raise ValueError(f'Covariance matrix for mixture {i} is not positive definite')
            vik = -0.5 * eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            sl = slice(i * p, (i + 1) * p)
            vi_stack[sl, :] = vik
            vim_stack[sl] = vik @ m[i, :]
            mtk_stack[sl] = m[i, :]
            lvm[i] = np.log(w[i]) - 0.5 * np.sum(np.log(eigvals))
        ss_sep = np.sqrt(np.min([v[j, j, :].min() for j in range(p)])) * ssf / np.sqrt(p)
        ss_sep = np.full(p, ss_sep)

    sv = 0.01 * ss_sep

    for it in range(maxloop):
        pg0 = pg.copy()

        if not full:
            # Compute probabilities
            py = np.zeros((k, nx))
            for ik in range(k):
                diff = pg - m[ik, :]
                py[ik, :] = np.sum(diff ** 2 * vi[ik, :], axis=1) + lvm[ik]

            mx_val = np.max(py, axis=0)
            px = np.exp(py - mx_val[np.newaxis, :])
            ps = np.sum(px, axis=0)
            px = (px / ps[np.newaxis, :]).T  # (nx, k)

            # Fixed point update
            pxvim = px @ vim  # (nx, p)
            pxvi = px @ vi  # (nx, p)
            pgf = pxvim / pxvi
        else:
            # Full covariance version
            py = np.zeros((k, nx))
            for ik in range(k):
                sl = slice(ik * p, (ik + 1) * p)
                vik = vi_stack[sl, :]
                vim_k = vim_stack[sl]
                mk = m[ik, :]
                for j_pt in range(nx):
                    yi = pg[j_pt, :]
                    py[ik, j_pt] = (vik @ yi - vim_k) @ (yi - mk) + lvm[ik]

            mx_val = np.max(py, axis=0)
            px = np.exp(py - mx_val[np.newaxis, :])
            ps = np.sum(px, axis=0)
            px = (px / ps[np.newaxis, :]).T  # (nx, k)

            # Fixed point update
            vif = np.zeros((k, p * p))
            vimf = np.zeros((k, p))
            for ik in range(k):
                sl = slice(ik * p, (ik + 1) * p)
                vif[ik, :] = vi_stack[sl, :].ravel()
                vimf[ik, :] = vim_stack[sl]

            pxvif = px @ vif  # (nx, p*p)
            pxvimf = px @ vimf  # (nx, p)
            pgf = np.zeros((nx, p))
            for j_pt in range(nx):
                pgf[j_pt, :] = np.linalg.solve(pxvif[j_pt, :].reshape(p, p), pxvimf[j_pt, :])

        if it <= nfp:
            pg = pgf
        else:
            pg = pgf  # simplified: just use fixed point

        if np.all(np.abs(pg - pg0) < sv[np.newaxis, :]):
            break

        # Remove duplicate modes
        if nx > 1:
            keep = np.ones(nx, dtype=bool)
            for i_mode in range(nx):
                if not keep[i_mode]:
                    continue
                for j_mode in range(i_mode + 1, nx):
                    if not keep[j_mode]:
                        continue
                    if np.all(np.abs(pg[i_mode, :] - pg[j_mode, :]) < ss_sep):
                        keep[j_mode] = False
            if not np.all(keep):
                pg = pg[keep, :]
                nx = pg.shape[0]

    # Calculate log pdf at each mode
    pv_arr = np.zeros(nx)
    for j_pt in range(nx):
        lp_val, _, _, _ = v_gaussmixp(pg[j_pt:j_pt + 1, :], m, v, w)
        pv_arr[j_pt] = lp_val[0]

    # Sort by decreasing probability
    ix = np.argsort(-pv_arr)
    pv_arr = pv_arr[ix]
    pg = pg[ix, :]

    if n_modes < len(pv_arr):
        pg = pg[:n_modes, :]
        pv_arr = pv_arr[:n_modes]

    return mg, vg, pg, pv_arr
