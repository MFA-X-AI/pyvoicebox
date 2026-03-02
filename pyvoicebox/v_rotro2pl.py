"""V_ROTRO2PL - find plane and rotation angle of a rotation matrix."""

import numpy as np
from scipy.linalg import schur


def v_rotro2pl(r):
    """Find the plane and rotation angle of a rotation matrix.

    Parameters
    ----------
    r : array_like, shape (n, n)
        Rotation matrix.

    Returns
    -------
    u : ndarray, shape (n,)
        First orthonormal vector defining the rotation plane.
    v : ndarray, shape (n,)
        Second orthonormal vector or rotated u.
    t : float
        Rotation angle in radians (0 <= t <= pi).
    """
    r = np.asarray(r, dtype=float)
    n = r.shape[0]

    # scipy.linalg.schur returns (T, Z) where A = Z @ T @ Z^H
    # MATLAB's [q, e] = schur(r) returns Z then T (i.e., q=Z, e=T)
    e, q = schur(r, output='real')

    # Find the 2x2 block with the largest subdiagonal element
    # e(2:n+1:n^2) in MATLAB = subdiagonal elements
    sub_diag = np.abs(e[np.arange(1, n), np.arange(n - 1)])
    i = np.argmax(sub_diag)  # 0-indexed position of largest subdiag

    z = int(e[i + 1, i] < 0)

    if z == 0:
        uv = q[:, i:i + 2]
    else:
        uv = q[:, i + 1:i - 1:-1] if i > 0 else q[:, i + 1::-1]

    u = uv[:, 0]
    v_vec = uv[:, 1]
    t_val = np.arctan2(np.abs(e[i + 1, i]), e[i, i])

    return u, v_vec, t_val
