"""V_LDATRACE - LDA transform to maximize trace discriminant."""

import numpy as np
from scipy import linalg


def v_ldatrace(b, w=None, n=None, c=None):
    """Calculate an LDA transform to maximize trace discriminant.

    Parameters
    ----------
    b : ndarray
        Between-class covariance matrix (m, m).
    w : ndarray, optional
        Within-class covariance matrix (m, m). Default: identity.
    n : int, optional
        Number of columns in output matrix A. Default: m.
    c : ndarray, optional
        Pre-specified columns of A (m, r). Default: None.

    Returns
    -------
    a : ndarray
        Transformation matrix (m, n): y = a.T @ x.
    f : ndarray
        Incremental gain in discriminant for successive columns.
    B : ndarray
        Between-class covariance of y.
    W : ndarray
        Within-class covariance of y.
    """
    b = np.asarray(b, dtype=float)
    m = b.shape[0]

    if w is None:
        w = np.eye(m)
    else:
        w = np.asarray(w, dtype=float)

    if n is None:
        n = m

    r = 0
    if c is not None:
        c = np.asarray(c, dtype=float)
        r = c.shape[1]

    if r > 0:
        a = np.zeros((m, n))
        if n > r:
            g = linalg.cholesky(w, lower=False)  # upper triangular
            # null space of c'*inv(g')
            ginv = linalg.solve_triangular(g, np.eye(m))
            ct_ginv = c.T @ ginv.T
            _, sv, vt = linalg.svd(ct_ginv, full_matrices=True)
            # Null space: columns of V corresponding to zero singular values
            null_dim = m - min(r, m)
            v_null = vt[min(r, m):, :].T
            v = ginv @ v_null
            p_mat, l_mat, q_mat = linalg.svd(v.T @ b @ v, full_matrices=True)
            a[:, r:n] = v @ p_mat[:, :n - r]
            a[:, :r] = c
        else:
            a = c[:, :n]

        if n > 0:
            aw = a.T @ w @ a
            ari = a @ linalg.solve_triangular(
                linalg.qr(linalg.cholesky(aw, lower=False))[1],
                np.eye(n)
            )
            f = np.diag(ari.T @ b @ ari)
    else:
        # Use generalized eigenvalue decomposition
        eigenvalues, eigenvectors = linalg.eig(b, w)
        eigenvalues = np.real(eigenvalues)
        idx = np.argsort(-eigenvalues)
        a = np.real(eigenvectors[:, idx[:n]])
        f = eigenvalues[idx[:n]]

    B_out = a.T @ b @ a
    W_out = a.T @ w @ a

    return a, f, B_out, W_out
