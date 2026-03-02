"""V_ROTQRVEC - rotate vectors by quaternion."""

import numpy as np
from pyvoicebox.v_rotqr2ro import v_rotqr2ro


def v_rotqrvec(q, x):
    """Apply a quaternion rotation to a vector array.

    Parameters
    ----------
    q : array_like, shape (4,)
        Quaternion [w, x, y, z] (possibly unnormalized).
    x : array_like
        Array of 3D column vectors, shape (3n, ...).

    Returns
    -------
    y : ndarray
        Rotated vectors, same shape as x.
    """
    q = np.asarray(q, dtype=float)
    x = np.asarray(x, dtype=float)
    original_shape = x.shape

    r = v_rotqr2ro(q)  # (3, 3)
    y = r @ x.reshape(3, -1)
    return y.reshape(original_shape)
