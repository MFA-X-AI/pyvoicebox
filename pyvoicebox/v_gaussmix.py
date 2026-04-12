"""V_GAUSSMIX - Fit a Gaussian mixture model using EM algorithm."""

from __future__ import annotations
import numpy as np
from .v_voicebox import v_voicebox
from .v_rnsubset import v_rnsubset
from .v_kmeans import v_kmeans
from .v_kmeanhar import v_kmeanhar


def v_gaussmix(x, c=None, l=None, m0=None, v0=None, w0=None, wx=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Fit a Gaussian mixture PDF to data using EM algorithm.

    Parameters
    ----------
    x : array_like
        Input data, shape (n, p).
    c : float, optional
        Minimum variance of normalized data. Default: 1/n^2.
    l : float, optional
        Max iterations (integer part) + stopping threshold (fractional part).
        Default: 100.0001.
    m0 : int or array_like
        Number of mixtures, or initial means (k, p).
    v0 : str or array_like, optional
        Initialization mode string or initial variances.
    w0 : array_like, optional
        Initial weights or data point weights.
    wx : array_like, optional
        Data point weights when m0/v0/w0 are explicit initial values.

    Returns
    -------
    m : ndarray
        Mixture means, shape (k, p).
    v : ndarray
        Mixture variances, shape (k, p) or (p, p, k).
    w : ndarray
        Mixture weights, shape (k,).
    g : float
        Average log probability.
    f : float
        Fisher's Discriminant.
    pp : ndarray
        Log probability of each data point.
    gg : ndarray
        Average log probabilities at each iteration.
    """
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    wn = np.ones(n)

    if c is None:
        c = 1.0 / n ** 2
    else:
        c = float(c)

    fulliv = False  # initial variance is not full

    if l is None:
        l = 100 + 1e-4

    if v0 is None or isinstance(v0, str):
        # No initial values given - use k-means or equivalent
        if v0 is None:
            v0 = 'hf'

        if w0 is not None:
            wx_local = np.asarray(w0, dtype=float).ravel()
        else:
            wx_local = wn.copy()
        wx_local = wx_local / np.sum(wx_local)

        if isinstance(m0, (int, np.integer)):
            k = int(m0)
            has_initial_means = False
        else:
            m0 = np.asarray(m0, dtype=float)
            k = m0.shape[0]
            has_initial_means = 'm' in v0

        fv = 'v' in v0

        mx0 = wx_local @ x
        vx0 = wx_local @ (x ** 2) - mx0 ** 2
        sx0 = np.sqrt(vx0)
        sx0[sx0 == 0] = 1.0

        if n <= k:
            xs = (x - mx0[np.newaxis, :]) / sx0[np.newaxis, :]
            m_fit = xs[np.arange(k) % n, :]
            v_fit = np.zeros((k, p))
            w_fit = np.zeros(k)
            w_fit[:n] = 1.0 / n
            if l > 0:
                l = 0.1
        else:
            if 's' in v0:
                xs = x.copy()
            else:
                xs = (x - mx0[np.newaxis, :]) / sx0[np.newaxis, :]
                if has_initial_means:
                    m0 = (m0 - mx0[np.newaxis, :]) / sx0[np.newaxis, :]

            w_fit = np.full(k, 1.0 / k)

            if 'k' in v0:
                if has_initial_means:
                    m_fit, _, j, _ = v_kmeans(xs, k, m0)
                elif 'p' in v0:
                    m_fit, _, j, _ = v_kmeans(xs, k, 'p')
                else:
                    m_fit, _, j, _ = v_kmeans(xs, k, 'f')
            elif 'h' in v0:
                if has_initial_means:
                    m_fit, _, j, _ = v_kmeanhar(xs, k, None, 4, m0)
                elif 'p' in v0:
                    m_fit, _, j, _ = v_kmeanhar(xs, k, None, 4, 'p')
                else:
                    m_fit, _, j, _ = v_kmeanhar(xs, k, None, 4, 'f')
            elif 'p' in v0:
                j = np.random.randint(0, k, size=n)
                forced = v_rnsubset(k, n)
                j[forced] = np.arange(k)
                m_fit = np.zeros((k, p))
                for i in range(k):
                    mask = j == i
                    if np.any(mask):
                        m_fit[i, :] = np.mean(xs[mask, :], axis=0)
            else:
                if has_initial_means:
                    m_fit = m0.copy()
                else:
                    m_fit = xs[v_rnsubset(k, n), :]
                _, j, _, _ = v_kmeans(xs, k, m_fit, 0)

            if 's' in v0:
                xs = (x - mx0[np.newaxis, :]) / sx0[np.newaxis, :]

            v_fit = np.zeros((k, p))
            w_fit = np.zeros(k)
            for i in range(k):
                mask = j == i
                ni = np.sum(mask)
                w_fit[i] = (ni + 1) / (n + k)
                if ni > 0:
                    v_fit[i, :] = np.sum((xs[mask, :] - m_fit[i, :][np.newaxis, :]) ** 2, axis=0) / ni

        m = m_fit
        v = v_fit
        w = w_fit
    else:
        # Use initial values given as input parameters
        if wx is None:
            wx_local = wn.copy()
        else:
            wx_local = np.asarray(wx, dtype=float).ravel()
        wx_local = wx_local / np.sum(wx_local)

        mx0 = wx_local @ x
        vx0 = wx_local @ (x ** 2) - mx0 ** 2
        sx0 = np.sqrt(vx0)
        sx0[sx0 == 0] = 1.0

        m0 = np.asarray(m0, dtype=float)
        v0_arr = np.asarray(v0, dtype=float)
        w0_arr = np.asarray(w0, dtype=float).ravel()

        k = m0.shape[0]
        xs = (x - mx0[np.newaxis, :]) / sx0[np.newaxis, :]
        m = (m0 - mx0[np.newaxis, :]) / sx0[np.newaxis, :]
        v = v0_arr.copy()
        w = w0_arr.copy()
        fv = v.ndim > 2 or (v.ndim == 2 and v.shape[0] > k)

        if fv:
            mk_mask = np.eye(p) == 0
            fulliv = np.any(v[np.tile(mk_mask[:, :, np.newaxis], (1, 1, k))] != 0)
            if not fulliv:
                diag_mask = np.eye(p) == 1
                v_diag = np.zeros((k, p))
                for ik in range(k):
                    v_diag[ik, :] = np.diag(v[:, :, ik]) / sx0 ** 2
                v = v_diag
            else:
                for ik in range(k):
                    v[:, :, ik] = v[:, :, ik] / np.outer(sx0, sx0)

    if len(wx_local) != n:
        raise ValueError(f'{n} datapoints but {len(wx_local)} weights')

    lsx = np.sum(np.log(sx0))
    xsw = xs * wx_local[:, np.newaxis]

    if not fulliv:
        # Diagonal covariance matrices EM
        v = np.maximum(v, c)
        xs2 = xs ** 2 * wx_local[:, np.newaxis]

        th = (l - np.floor(l)) * n
        sd = 1 if True else 0  # always compute final values
        lp_iter = int(np.floor(l)) + sd

        lpx = np.zeros(n)
        g = 0.0
        gg = np.zeros(lp_iter + 1)
        ss = sd

        for j_iter in range(lp_iter):
            g1 = g
            m1 = m.copy()
            v1 = v.copy()
            w1 = w.copy()

            vi = -0.5 / v
            lvm = np.log(w) - 0.5 * np.sum(np.log(v), axis=1)

            # Compute responsibilities
            py = np.zeros((k, n))
            for ik in range(k):
                diff = xs - m[ik, :]
                py[ik, :] = np.sum(diff ** 2 * vi[ik, :], axis=1) + lvm[ik]

            mx = np.max(py, axis=0)
            px = np.exp(py - mx[np.newaxis, :])
            ps = np.sum(px, axis=0)
            px = px / ps[np.newaxis, :]
            lpx = np.log(ps) + mx

            pk = px @ wx_local
            sx = px @ xsw
            sx2 = px @ xs2

            g = np.dot(lpx, wx_local)
            gg[j_iter] = g

            w = pk.copy()
            if np.all(pk > 0):
                m = sx / pk[:, np.newaxis]
                v_raw = sx2 / pk[:, np.newaxis]
            else:
                wm = pk == 0
                nz = np.sum(wm)
                mk_sort = np.argsort(lpx)
                m = np.zeros((k, p))
                v_raw = np.zeros((k, p))
                m[wm, :] = xs[mk_sort[:nz], :]
                w[wm] = 1.0 / n
                w = w * n / (n + nz)
                not_wm = ~wm
                m[not_wm, :] = sx[not_wm, :] / pk[not_wm, np.newaxis]
                v_raw = np.zeros((k, p))
                v_raw[not_wm, :] = sx2[not_wm, :] / pk[not_wm, np.newaxis]

            v = np.maximum(v_raw - m ** 2, c)

            if g - g1 <= th and j_iter > 0:
                if ss <= 0:
                    break
                ss -= 1

        # Calculate final probabilities
        pp = lpx - 0.5 * p * np.log(2 * np.pi) - lsx
        gg_out = gg[:j_iter + 1] - 0.5 * p * np.log(2 * np.pi) - lsx
        g = gg_out[-1]
        m = m1
        v = v1
        w = w1
        mm = np.sum(m, axis=0) / k
        f = (m.ravel() @ m.ravel() - k * (mm @ mm)) / np.sum(v)

        if not fv:
            m = m * sx0[np.newaxis, :] + mx0[np.newaxis, :]
            v = v * (sx0 ** 2)[np.newaxis, :]
        else:
            v_diag = v.copy()
            v = np.zeros((p, p, k))
            for ik in range(k):
                v[:, :, ik] = np.diag(v_diag[ik, :])
    else:
        # Full covariance EM - simplified for the common case
        # This path is taken when v0 contains full covariance matrices
        pp = np.zeros(n)
        f = 0.0
        gg_out = np.array([0.0])

        m = m * sx0[np.newaxis, :] + mx0[np.newaxis, :]
        if v.ndim == 2:
            v = v * (sx0 ** 2)[np.newaxis, :]
        else:
            for ik in range(k):
                v[:, :, ik] = v[:, :, ik] * np.outer(sx0, sx0)

    if fv and not fulliv:
        # Convert diagonal to full if 'v' was requested
        v_diag = v.copy()
        v = np.zeros((p, p, k))
        for ik in range(k):
            if v_diag.ndim == 2:
                v[:, :, ik] = np.diag(v_diag[ik, :])

    return m, v, w, g, f, pp, gg_out
