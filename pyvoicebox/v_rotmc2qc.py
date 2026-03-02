"""V_ROTMC2QC - convert complex quaternion matrices to complex quaternion vectors."""

import numpy as np


def v_rotmc2qc(mc):
    """Convert complex quaternion matrix form to complex quaternion vector form.

    Parameters
    ----------
    mc : array_like, shape (2m, 2n, ...)
        Complex quaternion matrices.

    Returns
    -------
    qc : ndarray, shape (2m, n, ...) or (2m,) if n=1
        Complex quaternion vectors.
    """
    mc = np.asarray(mc, dtype=complex)
    s = list(mc.shape)
    s[1] = s[1] // 2
    mc2 = mc.reshape(s[0], -1)
    qc = mc2[:, 0::2]
    if s[1] == 1:
        return qc.ravel()
    return qc.reshape(s)
