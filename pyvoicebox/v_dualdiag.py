"""V_DUALDIAG - Simultaneous diagonalization of two Hermitian matrices."""

import numpy as np
from scipy.linalg import eig


def v_dualdiag(w, b):
    """Simultaneous diagonalization of two Hermitian matrices.

    Parameters
    ----------
    w : array_like, shape (n, n)
        Hermitian matrix.
    b : array_like, shape (n, n)
        Hermitian matrix.

    Returns
    -------
    a : ndarray, shape (n, n)
        Diagonalizing matrix.
    d : ndarray, shape (n,)
        Real diagonal elements: A'*B*A = diag(D).
    e : ndarray, shape (n,)
        Real diagonal elements: A'*W*A = diag(E).
    """
    w = np.asarray(w, dtype=complex if np.iscomplexobj(w) else float)
    b = np.asarray(b, dtype=complex if np.iscomplexobj(b) else float)
    n = w.shape[0]

    # Generalized eigendecomposition
    eigenvalues, eigenvectors = eig(b + b.conj().T, w + w.conj().T)

    a = eigenvectors

    if np.isrealobj(eigenvalues):
        eigenvalues = eigenvalues.real
        # Sort by absolute value descending
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        a = a[:, idx]
        e = np.real(np.diag(a.conj().T @ w @ a))
        d = np.real(np.diag(a.conj().T @ b @ a))
    else:
        d = a.conj().T @ b @ a
        e = a.conj().T @ w @ a

    return a, d, e
