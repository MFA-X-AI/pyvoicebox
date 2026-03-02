"""V_ROTRO2QR - convert 3x3 rotation matrix to real quaternion."""

import numpy as np


def _qr_normalize(q):
    """Force first non-zero coefficient of quaternion to be positive."""
    sgn = np.sign(np.array([8, 4, 2, 1]) @ np.sign(q))
    sgn[sgn == 0] = 1
    return q * sgn[np.newaxis, :]


def v_rotro2qr(r):
    """Convert 3x3 rotation matrix to real quaternion.

    Parameters
    ----------
    r : array_like, shape (3, 3) or (3, 3, N)
        Rotation matrix/matrices.

    Returns
    -------
    q : ndarray, shape (4,) or (4, N)
        Quaternion(s) [w, x, y, z].
    """
    r = np.asarray(r, dtype=float)
    original_shape = r.shape
    squeeze = (r.ndim == 2)

    # Reshape to (9, N) - MATLAB column-major vectorization of 3x3
    if r.ndim == 2:
        # Single 3x3 matrix: vectorize column-major
        rv = r.ravel(order='F').reshape(9, 1)
    else:
        n = int(np.prod(original_shape[2:]))
        rv = np.zeros((9, n))
        for i in range(n):
            rv[:, i] = r[:, :, i].ravel(order='F')

    nq = rv.shape[1]
    q = np.zeros((4, nq))

    # d = 1 + r(1,1) + r(2,2) + r(3,3) = 4*cos^2(t/2)
    # In column-major vectorized form: r[0]=r(1,1), r[4]=r(2,2), r[8]=r(3,3)
    d = 1.0 + rv[0, :] + rv[4, :] + rv[8, :]

    mm = d > 1  # rotation angle < 120 degrees
    if np.any(mm):
        s = np.sqrt(d[mm]) * 2.0
        # r(2,3)-r(3,2) = rv[5]-rv[7], r(3,1)-r(1,3) = rv[6]-rv[2], r(1,2)-r(2,1) = rv[1]-rv[3]
        q[1, mm] = (rv[5, mm] - rv[7, mm]) / s
        q[2, mm] = (rv[6, mm] - rv[2, mm]) / s
        q[3, mm] = (rv[1, mm] - rv[3, mm]) / s
        q[0, mm] = 0.25 * s

    if np.any(~mm):
        # r(1,1) > r(2,2) and r(1,1) > r(3,3)
        m = (rv[0, :] > rv[4, :]) & (rv[0, :] > rv[8, :]) & ~mm
        if np.any(m):
            s = np.sqrt(1.0 + rv[0, m] - rv[4, m] - rv[8, m]) * 2.0
            q[1, m] = 0.25 * s
            q[2, m] = (rv[1, m] + rv[3, m]) / s
            q[3, m] = (rv[6, m] + rv[2, m]) / s
            q[0, m] = (rv[5, m] - rv[7, m]) / s
            mm = mm | m

        # r(2,2) > r(3,3)
        m = (rv[4, :] > rv[8, :]) & ~mm
        if np.any(m):
            s = np.sqrt(1.0 + rv[4, m] - rv[0, m] - rv[8, m]) * 2.0
            q[1, m] = (rv[1, m] + rv[3, m]) / s
            q[2, m] = 0.25 * s
            q[3, m] = (rv[5, m] + rv[7, m]) / s
            q[0, m] = (rv[6, m] - rv[2, m]) / s
            mm = mm | m

        # Remaining
        m = ~mm
        if np.any(m):
            s = np.sqrt(1.0 + rv[8, m] - rv[0, m] - rv[4, m]) * 2.0
            q[1, m] = (rv[6, m] + rv[2, m]) / s
            q[2, m] = (rv[5, m] + rv[7, m]) / s
            q[3, m] = 0.25 * s
            q[0, m] = (rv[1, m] - rv[3, m]) / s

    # Check normalization
    norm_err = np.max(np.abs(np.sum(q ** 2, axis=0) - 1.0))
    if norm_err > 1e-8:
        raise ValueError('Input to v_rotro2qr must be a rotation matrix with det(r)=+1')

    q = _qr_normalize(q)

    if squeeze:
        return q.ravel()
    else:
        return q
