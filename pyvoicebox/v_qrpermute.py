"""V_QRPERMUTE - Transpose or permute a quaternion array."""

import numpy as np


def v_qrpermute(x, p=None):
    """Transpose or permute a quaternion array.

    Parameters
    ----------
    x : array_like, shape (4m, ...)
        Real quaternion array.
    p : array_like, optional
        New order of dimensions (0-based). Default transposes first two dims.

    Returns
    -------
    y : ndarray
        Permuted quaternion array.
    """
    x = np.asarray(x, dtype=float)
    s = list(x.shape)
    if p is None:
        # Default: transpose first two dimensions
        if len(s) == 1:
            return x.copy()
        p = list(range(len(s)))
        p[0], p[1] = p[1], p[0]

    p = list(p)
    s[0] = s[0] // 4
    t = [s[i] for i in p]
    t[0] = 4 * t[0]

    # Reshape to (4, s0, s1, ...), permute, then reshape back
    new_shape = [4] + s
    perm = [0] + [i + 1 for i in p]
    y = np.transpose(x.reshape(new_shape), perm).reshape(t)
    return y
