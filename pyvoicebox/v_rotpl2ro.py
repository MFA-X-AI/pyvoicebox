"""V_ROTPL2RO - find rotation matrix from plane vectors."""

from __future__ import annotations
import numpy as np
from pyvoicebox.v_atan2sc import v_atan2sc


def v_rotpl2ro(u, v, t=None) -> np.ndarray:
    """Find rotation matrix to rotate in the plane containing u and v.

    Parameters
    ----------
    u : array_like, shape (n,)
        First vector defining the rotation plane.
    v : array_like, shape (n,)
        Second vector defining the rotation plane.
    t : float, optional
        Rotation angle in radians. If omitted, defaults to the angle
        between u and v.

    Returns
    -------
    r : ndarray, shape (n, n)
        Rotation matrix.
    """
    u = np.asarray(u, dtype=float).ravel()
    n = len(u)
    v = np.asarray(v, dtype=float).ravel()

    l = np.sqrt(u @ u)
    if l == 0:
        raise ValueError('input u is a zero vector')
    u = u / l

    q = v - (v @ u) * u  # q is orthogonal to u
    l = np.sqrt(q @ q)
    if l == 0:
        # u and v are colinear or v=zero
        _, i = np.max(np.abs(u)), np.argmax(np.abs(u))
        q = np.zeros(n)
        q[(i + 1) % n] = 1.0
        q = q - (q @ u) * u
        l = np.sqrt(q @ q)
    q = q / l

    if t is None:
        s, c, _, _ = v_atan2sc(v @ q, v @ u)
        r = (np.eye(n) + (c - 1) * (np.outer(u, u) + np.outer(q, q))
             + s * (np.outer(q, u) - np.outer(u, q)))
    else:
        r = (np.eye(n) + (np.cos(t) - 1) * (np.outer(u, u) + np.outer(q, q))
             + np.sin(t) * (np.outer(q, u) - np.outer(u, q)))
    return r
