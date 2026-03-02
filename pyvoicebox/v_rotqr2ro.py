"""V_ROTQR2RO - convert real quaternion to 3x3 rotation matrix."""

import numpy as np

# Precomputed index arrays (from MATLAB source, converted to 0-indexed)
_A = np.array([0, 4, 8])       # diagonal
_B = np.array([10, 15, 5])     # indices into 16-element quadratic terms
_C = np.array([15, 5, 10])
_D = np.array([3, 7, 2])       # off-diagonal (lower)
_E = np.array([9, 14, 13])
_F = np.array([3, 1, 2])
_G = np.array([1, 5, 6])       # off-diagonal (upper)
_H = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
_M = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])


def v_rotqr2ro(q):
    """Convert real quaternion to 3x3 rotation matrix.

    Parameters
    ----------
    q : array_like, shape (4,) or (4, N)
        Real quaternion(s) [w, x, y, z].

    Returns
    -------
    r : ndarray, shape (3, 3) or (3, 3, N)
        Rotation matrix/matrices.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q.reshape(4, 1)
        squeeze = True
    else:
        squeeze = False

    nq = q.shape[1]

    # Compute quadratic terms: p = 2*q[h]*q[m] / sum(q^2)
    qnorm = np.sum(q ** 2, axis=0, keepdims=True)  # (1, nq)
    p = 2.0 * q[_H, :] * q[_M, :] / np.tile(qnorm, (16, 1))  # (16, nq)

    r = np.zeros((9, nq))
    r[_A, :] = 1.0 - p[_B, :] - p[_C, :]
    r[_D, :] = p[_E, :] - p[_F, :]
    r[_G, :] = p[_E, :] + p[_F, :]

    if squeeze:
        return r[:, 0].reshape(3, 3, order='F')
    else:
        result = np.zeros((3, 3, nq))
        for i in range(nq):
            result[:, :, i] = r[:, i].reshape(3, 3, order='F')
        return result
