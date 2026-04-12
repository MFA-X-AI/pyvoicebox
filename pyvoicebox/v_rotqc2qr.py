"""V_ROTQC2QR - convert complex quaternion to real quaternion."""

from __future__ import annotations
import numpy as np


def v_rotqc2qr(qc) -> np.ndarray:
    """Convert complex quaternion [r+j*b, a+j*c] to real [r, a, b, c].

    Parameters
    ----------
    qc : array_like, shape (2m, ...)
        Complex quaternion vectors.

    Returns
    -------
    qr : ndarray, shape (4m, ...)
        Real quaternion vectors.
    """
    qc = np.asarray(qc, dtype=complex)
    s = list(qc.shape)
    s[0] = 2 * s[0]

    # Interleave real and imaginary parts: [real, imag] for each element
    flat = qc.ravel()
    pairs = np.array([flat.real, flat.imag])  # (2, total)
    qr = pairs.reshape(4, -1)  # (4, total/2)
    # Reorder from [r, b, a, c] to [r, a, b, c]
    a = np.array([0, 2, 1, 3])
    qr = qr[a, :]
    qr = qr.reshape(s)
    return qr
