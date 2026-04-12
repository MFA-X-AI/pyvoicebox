"""V_ROTAX2QR - convert rotation axis and angle to quaternion."""

from __future__ import annotations
import numpy as np


def v_rotax2qr(a, t) -> np.ndarray:
    """Convert rotation axis and angle to quaternion.

    Parameters
    ----------
    a : array_like, shape (3,)
        Rotation axis vector (need not be unit length).
    t : float
        Rotation angle in radians.

    Returns
    -------
    q : ndarray, shape (4,)
        Quaternion [w, x, y, z].
    """
    a = np.asarray(a, dtype=float).ravel()
    if np.all(a == 0):
        a = np.array([1.0, 0.0, 0.0])
    m = np.sqrt(a @ a)
    q = np.zeros(4)
    q[0] = np.cos(0.5 * t)
    q[1:] = np.sin(0.5 * t) * a / m
    return q
