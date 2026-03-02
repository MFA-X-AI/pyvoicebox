"""V_GAUSSMIXT - Multiply two GMM PDFs."""

import numpy as np


def v_gaussmixt(m1, v1, w1, m2, v2, w2):
    """Multiply two GMM PDFs.

    Parameters
    ----------
    m1, m2 : array_like
        Mixture means, shape (k1, p) and (k2, p).
    v1, v2 : array_like
        Variances: diagonal (ki, p) or full (p, p, ki).
    w1, w2 : array_like
        Mixture weights.

    Returns
    -------
    m : ndarray
        Product mixture means, shape (k1*k2, p).
    v : ndarray
        Product variances.
    w : ndarray
        Product weights, shape (k1*k2,).
    """
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    w1 = np.asarray(w1, dtype=float).ravel()
    w2 = np.asarray(w2, dtype=float).ravel()

    if m1.ndim == 1:
        m1 = m1.reshape(1, -1)
    if m2.ndim == 1:
        m2 = m2.reshape(1, -1)

    k1, p = m1.shape
    k2 = m2.shape[0]
    f1 = v1.ndim > 2 or (v1.ndim == 2 and v1.shape[0] > k1)
    f2 = v2.ndim > 2 or (v2.ndim == 2 and v2.shape[0] > k2)

    k = k1 * k2
    j1 = np.tile(np.arange(k1), k2)
    j2 = np.repeat(np.arange(k2), k1)

    if p == 1:
        p1 = 1.0 / v1.ravel()
        p2 = 1.0 / v2.ravel()
        v = 1.0 / (p1[j1] + p2[j2])
        s1 = p1 * m1.ravel()
        s2 = p2 * m2.ravel()
        m = v * (s1[j1] + s2[j2])
        v12 = v1.ravel()[j1] + v2.ravel()[j2]
        wx = -0.5 * (m1.ravel()[j1] - m2.ravel()[j2]) ** 2 / v12
        wx = wx - np.max(wx)
        w = w1[j1] * w2[j2] * np.exp(wx) / np.sqrt(v12)
        w = w / np.sum(w)
        m = m.reshape(-1, 1)
        v = v.reshape(-1, 1)
    elif not f1 and not f2:
        # Both diagonal
        p1 = 1.0 / v1
        p2 = 1.0 / v2
        v = 1.0 / (p1[j1, :] + p2[j2, :])
        s1 = p1 * m1
        s2 = p2 * m2
        m = v * (s1[j1, :] + s2[j2, :])
        v12 = v1[j1, :] + v2[j2, :]
        wx = -0.5 * np.sum((m1[j1, :] - m2[j2, :]) ** 2 / v12, axis=1)
        wx = wx - np.max(wx)
        w = w1[j1] * w2[j2] * np.exp(wx) / np.sqrt(np.prod(v12, axis=1))
        w = w / np.sum(w)
    else:
        # At least one full covariance
        m = np.zeros((k, p))
        v = np.zeros((p, p, k))
        w = np.zeros(k)
        wx = np.zeros(k)

        for idx in range(k):
            i = j1[idx]
            j = j2[idx]

            if f1:
                v1i = v1[:, :, i]
                p1i = np.linalg.inv(v1i)
            else:
                v1i = np.diag(v1[i, :])
                p1i = np.diag(1.0 / v1[i, :])

            if f2:
                v2j = v2[:, :, j]
                p2j = np.linalg.inv(v2j)
            else:
                v2j = np.diag(v2[j, :])
                p2j = np.diag(1.0 / v2[j, :])

            pij = p1i + p2j
            vix = np.linalg.inv(pij)
            vij = v1i + v2j

            v[:, :, idx] = vix
            pm1 = m1[i, :] @ p1i
            pm2 = m2[j, :] @ p2j
            m[idx, :] = (pm1 + pm2) @ vix

            m12 = m1[i, :] - m2[j, :]
            wx[idx] = -0.5 * m12 @ np.linalg.solve(vij, m12)
            w[idx] = w1[i] * w2[j] / np.sqrt(np.linalg.det(vij))

        wx = wx - np.max(wx)
        w = w * np.exp(wx)
        w = w / np.sum(w)
        if k == 1:
            v = v[:, :, 0]

    return m, v, w
