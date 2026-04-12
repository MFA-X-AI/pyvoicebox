"""V_ROTQR2AX - convert quaternion to rotation axis and angle."""

from __future__ import annotations
import numpy as np


def v_rotqr2ax(q) -> tuple[np.ndarray, float]:
    """Convert quaternion to rotation axis and angle.

    Parameters
    ----------
    q : array_like, shape (4,)
        Quaternion [w, x, y, z].

    Returns
    -------
    a : ndarray, shape (3,)
        Unit rotation axis vector.
    t : float
        Rotation angle in radians (0 to 2*pi).
    """
    q = np.asarray(q, dtype=float).ravel()
    a = q[1:4].copy()
    m = np.sqrt(a @ a)
    t = 2.0 * np.arctan2(m, q[0])
    if m > 0:
        a = a / m
    else:
        a = np.array([0.0, 0.0, 1.0])
    return a, t
