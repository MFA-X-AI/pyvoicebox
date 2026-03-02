"""V_ROTEU2QR - convert Euler angles to real unit quaternion."""

import numpy as np
from pyvoicebox.v_roteucode import v_roteucode

# Precomputed index table (from MATLAB source)
# y is (12, 8): for each of 12 rotation codes (3 axes x 4 quadrants), 8 indices
_Y = np.tile(np.array([
    [2, 4, 1, 3, 1, 3, 2, 4],
    [3, 2, 1, 4, 1, 4, 3, 2],
    [3, 4, 2, 1, 1, 2, 4, 3],
], dtype=int), (4, 1))

_CB = np.cos(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]) * np.pi / 4)
_SB = np.sin(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]) * np.pi / 4)


def _qr_normalize(q):
    """Force first non-zero coefficient of quaternion to be positive."""
    # q is (4, n)
    sgn = np.sign(np.array([8, 4, 2, 1]) @ np.sign(q))
    sgn[sgn == 0] = 1
    return q * sgn[np.newaxis, :]


def v_roteu2qr(m, e=None):
    """Convert Euler angles to real unit quaternion.

    Parameters
    ----------
    m : str
        Rotation code string.
    e : array_like, optional
        Euler angles. Shape (K,) or (K, N) where K is the number of
        xyz rotations in m.

    Returns
    -------
    q : ndarray, shape (4,) or (4, N)
        Quaternion(s) [w, x, y, z].
    """
    mv = v_roteucode(m)
    ne = int(mv[1, -1])  # number of euler angles needed

    if ne == 0:
        q = np.zeros((4, 1))
        q[0, :] = 1.0
        r = q.copy()
    else:
        e = np.asarray(e, dtype=float)
        original_shape = e.shape
        if e.ndim == 1:
            e = e.reshape(-1, 1)
        elif e.ndim == 2 and original_shape[0] == 1 and e.size == ne:
            e = e.reshape(-1, 1)
        else:
            e = e.reshape(original_shape[0], -1)
        q = np.zeros((4, e.shape[1]))
        q[0, :] = 1.0
        r = q.copy()

    ef = 0.5 * mv[3, -1]  # signed euler angle scale factor (includes 0.5)
    nm = mv.shape[1] - 1  # number of rotations

    for i in range(nm):
        mvi = mv[:, i]
        mi = int(mvi[0])  # 1-indexed rotation code
        x = _Y[mi - 1, :] - 1  # convert to 0-indexed

        if mi <= 3:
            ei = int(mvi[1]) - 1  # 0-indexed euler angle index
            b = ef * e[ei, :]  # semi-rotation angle in radians
            c = np.cos(b)
            s = np.sin(b)
        else:
            c = _CB[mi - 1]
            s = _SB[mi - 1]

        r[x[0:2], :] = q[x[2:4], :]
        r[x[4:6], :] = -q[x[6:8], :]
        if np.isscalar(c):
            q = c * q + s * r
        else:
            q = c[np.newaxis, :] * q + s[np.newaxis, :] * r

    q[0, :] *= mv[6, -1]  # invert rotation if necessary

    q = _qr_normalize(q)

    # Reshape output
    if ne == 0:
        return q.ravel()
    elif e.shape[1] == 1:
        return q.ravel()
    else:
        return q
