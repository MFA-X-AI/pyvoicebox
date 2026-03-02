"""V_DLYAPSQ - Solve discrete Lyapunov equation in square root form."""

import numpy as np
from scipy.linalg import schur, rsf2csf, qr


def v_dlyapsq(a, b):
    """Solve the discrete Lyapunov equation AV'VA' - V'V + BB' = 0.

    V is upper triangular with real non-negative diagonal entries.
    Equivalent to chol(dlyap(a, b@b')) but better conditioned.

    Parameters
    ----------
    a : array_like, shape (n, n)
        State matrix.
    b : array_like, shape (n, m)
        Input matrix.

    Returns
    -------
    v : ndarray, shape (n, n)
        Upper triangular solution.
    """
    a = np.asarray(a, dtype=complex if np.iscomplexobj(a) else float)
    b = np.asarray(b, dtype=complex if np.iscomplexobj(b) else float)

    # Schur decomposition of a'
    s, q = schur(a.conj().T)
    s, q = rsf2csf(s, q)

    # QR factorization of b'*q
    r = np.linalg.qr(b.conj().T @ q, mode='r')

    m, n = r.shape if r.ndim == 2 else (1, r.shape[0])
    if r.ndim == 1:
        r = r.reshape(1, -1)

    u = np.zeros((n, n), dtype=complex)

    if m == 1:
        for i in range(n - 1):
            si = s[i, i]
            aa = np.sqrt(1 - si * np.conj(si))
            u[i, i] = r[0, 0] / aa
            rhs = u[i, i] * np.conj(si) * s[i, i + 1:] + aa * r[0, 1:]
            lhs = np.eye(n - i - 1) - np.conj(si) * s[i + 1:, i + 1:]
            u[i, i + 1:] = np.linalg.solve(lhs, rhs)
            r = np.atleast_2d(aa * (u[i, i] * s[i, i + 1:] + u[i, i + 1:] @ s[i + 1:, i + 1:]) - si * r[0, 1:])
        u[n - 1, n - 1] = r[0, 0] / np.sqrt(1 - s[n - 1, n - 1] * np.conj(s[n - 1, n - 1]))
    else:
        # General case with m > 1
        w = np.zeros(m)
        w[m - 1] = 1.0
        em = np.eye(m)
        for i in range(n - m):
            si = s[i, i]
            aa = np.sqrt(1 - si * np.conj(si))
            u[i, i] = r[0, 0] / aa
            rhs = u[i, i] * np.conj(si) * s[i, i + 1:] + aa * r[0, 1:]
            lhs = np.eye(n - i - 1) - np.conj(si) * s[i + 1:, i + 1:]
            u[i, i + 1:] = np.linalg.solve(lhs, rhs)
            vv = aa * (u[i, i] * s[i, i + 1:] + u[i, i + 1:] @ s[i + 1:, i + 1:]) - si * r[0, 1:]
            rr = np.zeros((m, n - i - 1), dtype=complex)
            rr[:m - 1, :] = r[1:, 1:]
            # QR update equivalent
            combined = np.vstack([rr, vv.reshape(1, -1)])
            qq, r_new = np.linalg.qr(combined, mode='reduced')
            r = r_new[:m, :]

        for i in range(max(0, n - m), n - 1):
            si = s[i, i]
            aa = np.sqrt(1 - si * np.conj(si))
            u[i, i] = r[0, 0] / aa
            rhs = u[i, i] * np.conj(si) * s[i, i + 1:] + aa * r[0, 1:]
            lhs = np.eye(n - i - 1) - np.conj(si) * s[i + 1:, i + 1:]
            u[i, i + 1:] = np.linalg.solve(lhs, rhs)
            vv = aa * (u[i, i] * s[i, i + 1:] + u[i, i + 1:] @ s[i + 1:, i + 1:]) - si * r[0, 1:]
            ni = n - i - 1
            rr = np.zeros((ni + 1, ni), dtype=complex)
            rr[:ni, :] = r[1:ni + 1, 1:] if r.shape[0] > 1 else np.zeros((ni, ni), dtype=complex)
            combined = np.vstack([rr[:ni, :], vv.reshape(1, -1)])
            qq, r_new = np.linalg.qr(combined, mode='reduced')
            r = r_new[:ni, :]

        u[n - 1, n - 1] = r.ravel()[0] / np.sqrt(1 - s[n - 1, n - 1] * np.conj(s[n - 1, n - 1]))

    v = np.triu(np.linalg.qr(u @ q.conj().T, mode='r'))
    dv = np.diag(v)
    ix = dv != 0
    signs = np.ones(n, dtype=complex)
    signs[ix] = np.abs(dv[ix]) / dv[ix]
    v = np.diag(signs) @ v

    if np.isrealobj(a) and np.isrealobj(b):
        v = np.real(v)
    return v
