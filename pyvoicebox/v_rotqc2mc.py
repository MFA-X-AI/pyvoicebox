"""V_ROTQC2MC - convert complex quaternion vectors to complex quaternion matrices."""

from __future__ import annotations
import numpy as np


def v_rotqc2mc(qc) -> np.ndarray:
    """Convert complex quaternion vector form to complex quaternion matrix form.

    Each quaternion r+ai+bj+ck is stored as a 2x1 complex vector [r+bi, a+ci].
    The matrix form is [[r+bi, -a+ci], [a+ci, r-bi]].

    Parameters
    ----------
    qc : array_like, shape (2m,) or (2m, n, ...)
        Complex quaternion vectors.

    Returns
    -------
    mc : ndarray, shape (2m, 2) or (2m, 2n, ...)
        Complex quaternion matrices.
    """
    qc = np.asarray(qc, dtype=complex)
    squeeze = (qc.ndim == 1)
    if squeeze:
        qc = qc.reshape(-1, 1)

    s = list(qc.shape)
    m = s[0]
    qa = qc.reshape(m, -1)
    n_cols = qa.shape[1]
    mc = np.zeros((m, 2 * n_cols), dtype=complex)

    ix = np.arange(0, m, 2)  # even rows (0-indexed)

    # Odd columns (0-indexed: 0, 2, 4, ...) = qa
    mc[:, 0::2] = qa
    # Even columns (1, 3, 5, ...)
    mc[ix, 1::2] = -np.conj(qa[ix + 1, :])
    mc[ix + 1, 1::2] = np.conj(qa[ix, :])

    if not squeeze:
        s[1] = 2 * s[1]
        mc = mc.reshape(s)
    return mc
