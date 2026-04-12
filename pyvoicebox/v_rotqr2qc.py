"""V_ROTQR2QC - convert real quaternion to complex quaternion."""

from __future__ import annotations
import numpy as np


def v_rotqr2qc(qr) -> np.ndarray:
    """Convert real quaternion [r, a, b, c] to complex [r+j*b, a+j*c].

    Parameters
    ----------
    qr : array_like, shape (4m, ...)
        Real quaternion vectors.

    Returns
    -------
    qc : ndarray, shape (2m, ...)
        Complex quaternion vectors.
    """
    qr = np.asarray(qr, dtype=float)
    s = list(qr.shape)
    qq = qr.reshape(4, -1)  # (4, n)
    # Reorder: [r, a, b, c] -> [r, b, a, c]
    a = np.array([0, 2, 1, 3])
    qq = qq[a, :]
    # Now [r, b, a, c]: pair (r, b) -> r + j*b, (a, c) -> a + j*c
    qc = qq[0::2, :] + 1j * qq[1::2, :]  # (2, n)
    s[0] = s[0] // 2
    qc = qc.reshape(s)
    return qc
