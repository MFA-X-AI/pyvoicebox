"""V_ROTQR2MR - convert real quaternion vectors to quaternion matrices."""

from __future__ import annotations
import numpy as np


def v_rotqr2mr(qr) -> np.ndarray:
    """Convert real quaternion vector form to real quaternion matrix form.

    Each quaternion [w, x, y, z] becomes a 4x4 matrix:
    [[ w, -x, -y, -z],
     [ x,  w, -z,  y],
     [ y,  z,  w, -x],
     [ z, -y,  x,  w]]

    Parameters
    ----------
    qr : array_like, shape (4m,) or (4m, n, ...)
        Real quaternion vectors.

    Returns
    -------
    mr : ndarray, shape (4m, 4) or (4m, 4n, ...)
        Real quaternion matrices.
    """
    qr = np.asarray(qr, dtype=float)
    squeeze = (qr.ndim == 1)
    if squeeze:
        qr = qr.reshape(-1, 1)

    s = list(qr.shape)
    m = s[0]
    qa = qr.reshape(m, -1)
    n_total = qa.shape[1]
    n_quats = m // 4

    mr = np.zeros((m, 4 * n_total))

    for qi in range(n_quats):
        for col_idx in range(n_total):
            w = qa[4 * qi + 0, col_idx]
            x = qa[4 * qi + 1, col_idx]
            y = qa[4 * qi + 2, col_idx]
            z = qa[4 * qi + 3, col_idx]

            row_base = 4 * qi
            col_base = 4 * col_idx

            mr[row_base:row_base + 4, col_base:col_base + 4] = np.array([
                [w, -x, -y, -z],
                [x, w, -z, y],
                [y, z, w, -x],
                [z, -y, x, w],
            ])

    if squeeze:
        return mr
    else:
        out_s = list(s)
        out_s[1] = 4 * s[1]
        return mr.reshape(out_s)
