"""V_QRDOTMULT - Element-wise quaternion multiplication."""

from __future__ import annotations
import numpy as np


def v_qrdotmult(q1, q2) -> np.ndarray:
    """Multiply two real quaternion arrays element-wise (Hadamard product).

    Parameters
    ----------
    q1 : array_like, shape (4n, ...)
        First quaternion array.
    q2 : array_like, shape (4n, ...)
        Second quaternion array.

    Returns
    -------
    q : ndarray, shape (4n, ...)
        Element-wise quaternion product.
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    s = q1.shape
    # Use Fortran order to match MATLAB column-major reshape
    qa = q1.reshape(4, -1, order='F')
    qb = q2.reshape(4, -1, order='F')

    # Index arrays (0-based)
    a = np.array([0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0])
    b = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    # Sign negation indices (0-based from MATLAB c=[2 3 4 7 12 14] -> 1,2,3,6,11,13)
    c = np.array([1, 2, 3, 6, 11, 13])

    prod = qa[a, :] * qb[b, :]
    prod[c, :] = -prod[c, :]
    # MATLAB: reshape(sum(reshape(q,4,[]),1),s) -- all column-major
    q = prod.reshape(4, -1, order='F').sum(axis=0).reshape(s, order='F')
    return q
