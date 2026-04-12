"""V_QRDIVIDE - Divide two real quaternions."""

from __future__ import annotations
import numpy as np


def v_qrdivide(q1, q2=None) -> np.ndarray:
    """Divide two real quaternions: q = q1/q2 such that q1 = q*q2.

    Parameters
    ----------
    q1 : array_like, shape (4,)
        First quaternion [r, i, j, k].
    q2 : array_like, shape (4,), optional
        Second quaternion. If omitted, returns inverse of q1.

    Returns
    -------
    q : ndarray, shape (4,)
        Quotient quaternion.
    """
    q1 = np.asarray(q1, dtype=float).ravel()

    if q2 is None:
        # Just invert q1
        q = q1 / (q1 @ q1)
        q[1:4] = -q[1:4]
        return q

    q2 = np.asarray(q2, dtype=float).ravel()

    # Invert q2
    qi = q2 / (q2 @ q2)
    qi[1:4] = -qi[1:4]

    # Multiply q1 * qi
    t = np.outer(q1, qi)
    s = np.zeros((4, 4))

    # MATLAB indices (1-based): a=[5 8 9 10 15 13], b=[6 7 11 12 14 16]
    # These are linearized indices into a 4x4 matrix (column-major)
    # a: (0,1),(3,1),(0,2),(1,2),(2,3),(0,3) -> negate t at b positions
    # b: (1,1),(2,1),(2,2),(3,2),(1,3),(3,3)

    # c=[1 2 3 4 6 7 11 12 16 14] (1-based, col-major) -> positive terms
    # d=[1 2 3 4 5 8 9 10 13 15] (1-based, col-major)

    # Direct quaternion multiplication: q1 * qi
    r1, i1, j1, k1 = q1
    r2, i2, j2, k2 = qi

    q = np.zeros(4)
    q[0] = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2
    q[1] = r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2
    q[2] = r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2
    q[3] = r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2
    return q
