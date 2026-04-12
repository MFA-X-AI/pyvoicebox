"""V_ROTMR2QR - convert real quaternion matrices to quaternion vectors."""

from __future__ import annotations
import numpy as np


def v_rotmr2qr(mr) -> np.ndarray:
    """Convert real quaternion matrix form to real quaternion vector form.

    Parameters
    ----------
    mr : array_like, shape (4m, 4n, ...)
        Real quaternion matrices.

    Returns
    -------
    qr : ndarray, shape (4m,) or (4m, n, ...)
        Real quaternion vectors.
    """
    mr = np.asarray(mr, dtype=float)
    s = list(mr.shape)
    n_out = s[1] // 4
    mr2 = mr.reshape(s[0], -1)
    qr = mr2[:, 0::4]
    if n_out == 1:
        return qr.ravel()
    else:
        out_s = list(s)
        out_s[1] = n_out
        return qr.reshape(out_s)
