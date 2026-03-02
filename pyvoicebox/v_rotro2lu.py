"""V_ROTRO2LU - convert rotation matrix to look and up directions."""

import numpy as np


def v_rotro2lu(r):
    """Convert rotation matrix to look and up directions.

    The rotation maps the look direction to the negative z-axis and the
    up direction to lie in the y-z plane with a positive y component.

    Parameters
    ----------
    r : array_like, shape (3, 3) or (3, 3, N)
        Rotation matrix/matrices.

    Returns
    -------
    l : ndarray, shape (3,) or (3, N)
        Look direction vector(s).
    u : ndarray, shape (3,) or (3, N)
        Up direction vector(s).
    """
    r = np.asarray(r, dtype=float)
    original_shape = r.shape

    if r.ndim == 2:
        # Single rotation matrix
        rv = r.ravel(order='F')  # column-major vectorization
        l = -rv[[2, 5, 8]]
        u = rv[[1, 4, 7]]
    else:
        n = int(np.prod(original_shape[2:]))
        rv = np.zeros((9, n))
        for i in range(n):
            rv[:, i] = r[:, :, i].ravel(order='F')
        out_shape = [3] + list(original_shape[2:])
        l = -rv[[2, 5, 8], :].reshape(out_shape)
        u = rv[[1, 4, 7], :].reshape(out_shape)

    return l, u
