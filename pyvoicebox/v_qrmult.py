"""V_QRMULT - Multiply two real quaternion matrices."""

import numpy as np
from pyvoicebox.v_qrdotmult import v_qrdotmult
from pyvoicebox.v_rotqr2mr import v_rotqr2mr


def v_qrmult(q1, q2):
    """Multiply two real quaternion matrices.

    Parameters
    ----------
    q1 : array_like, shape (4m, n)
        First quaternion matrix.
    q2 : array_like, shape (4n, r)
        Second quaternion matrix.

    Returns
    -------
    q : ndarray, shape (4m, r)
        Matrix product.
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    s1 = q1.shape
    s2 = q2.shape

    if s1 == (4,) or (len(s1) == 2 and s1 == (4, 1)):
        # q1 is a scalar quaternion
        q1c = q1.reshape(4, 1) if q1.ndim == 1 else q1
        q2c = q2 if q2.ndim == 2 else q2.reshape(4, 1)
        return v_qrdotmult(np.tile(q1c, (q2c.shape[0] // 4, q2c.shape[1])), q2c)
    elif s2 == (4,) or (len(s2) == 2 and s2 == (4, 1)):
        # q2 is a scalar quaternion
        q2c = q2.reshape(4, 1) if q2.ndim == 1 else q2
        q1c = q1 if q1.ndim == 2 else q1.reshape(4, 1)
        return v_qrdotmult(q1c, np.tile(q2c, (q1c.shape[0] // 4, q1c.shape[1])))
    else:
        return v_rotqr2mr(q1) @ q2
