"""V_QRABS - Absolute value and normalization of real quaternions."""

from __future__ import annotations
import numpy as np


def v_qrabs(q1) -> tuple[np.ndarray, np.ndarray]:
    """Absolute value and normalization of real quaternions.

    Parameters
    ----------
    q1 : array_like, shape (4n, ...)
        Real quaternion array.

    Returns
    -------
    m : ndarray, shape (n, ...)
        Quaternion magnitudes.
    q : ndarray, shape (4n, ...)
        Normalized quaternions (unit magnitude).
    """
    q1 = np.asarray(q1, dtype=float)
    s = list(q1.shape)
    q = q1.reshape(4, -1).copy()
    m = np.sqrt(np.sum(q ** 2, axis=0))

    # Normalize non-zero quaternions
    nz = m > 0
    q[:, nz] = q[:, nz] / m[np.newaxis, nz]
    # Zero quaternions become [1, 0, 0, 0]
    q[0, ~nz] = 1.0
    q[1:, ~nz] = 0.0

    q = q.reshape(s)
    s[0] = s[0] // 4
    m = m.reshape(s)
    return m, q
