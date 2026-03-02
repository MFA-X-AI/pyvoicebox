"""V_QRDOTDIV - Element-wise quaternion division."""

import numpy as np


def v_qrdotdiv(x, y=None):
    """Divide two real quaternion arrays element-wise.

    Parameters
    ----------
    x : array_like, shape (4n, ...)
        First quaternion array.
    y : array_like, shape (4n, ...), optional
        Second quaternion array. If omitted, returns inverse of x.

    Returns
    -------
    q : ndarray, shape (4n, ...)
        Element-wise quaternion quotient.
    """
    x = np.asarray(x, dtype=float)
    s = x.shape
    # Use Fortran order to match MATLAB column-major reshape
    xr = x.reshape(4, -1, order='F')

    if y is None:
        # Invert x
        m = np.sum(xr ** 2, axis=0)
        result = xr / m[np.newaxis, :]
        result[1:, :] = -result[1:, :]
        return result.reshape(s, order='F')

    y = np.asarray(y, dtype=float)
    yr = y.reshape(4, -1, order='F')
    m = np.sum(yr ** 2, axis=0)  # shape (N,)

    # Index arrays (0-based)
    a = np.array([0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0])
    b = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    # Sign negation indices (0-based from MATLAB c=[6 8 10 11 15 16] -> 5,7,9,10,14,15)
    c = np.array([5, 7, 9, 10, 14, 15])

    prod = xr[a, :] * yr[b, :]
    prod[c, :] = -prod[c, :]
    # MATLAB: reshape(sum(reshape(q,4,[]),1),s)./m(ones(4,1),:)
    # After sum, get (4N,) reshaped to s. Divide by m replicated to match.
    q_4xN = prod.reshape(4, -1, order='F').sum(axis=0).reshape(4, -1, order='F')
    q_4xN = q_4xN / m[np.newaxis, :]
    q = q_4xN.reshape(s, order='F')
    return q
