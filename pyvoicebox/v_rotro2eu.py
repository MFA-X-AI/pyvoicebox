"""V_ROTRO2EU - convert rotation matrix to Euler angles."""

import numpy as np
from pyvoicebox.v_roteucode import v_roteucode, ZEL, MES, _TRMAP
from pyvoicebox.v_atan2sc import v_atan2sc

# Precomputed constants (from MATLAB source)
_RTR = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])  # indices to transpose vectorized 3x3 (0-indexed)
_RTCI = np.array([
    [1, 2, 4, 5, 7, 8],
    [2, 0, 5, 3, 8, 6],
    [0, 1, 3, 4, 6, 7],
], dtype=int).T  # (6, 3), 0-indexed

_RTSI = np.array([
    [2, 1, 5, 4, 8, 7],
    [0, 2, 3, 5, 6, 8],
    [1, 0, 4, 3, 7, 6],
], dtype=int).T  # (6, 3), 0-indexed

# [sin; -sin; cos; xyz] for fixed rotations (codes 4-12)
# scai columns are indexed by rotation code (4-12, so 0-indexed as 3-11)
_SCAI = np.array([
    [0, 0, 0, 1],
    [0, 0, 0, 2],
    [0, 0, 0, 3],
    [1, -1, 0, 1],
    [1, -1, 0, 2],
    [1, -1, 0, 3],
    [0, 0, -1, 1],
    [0, 0, -1, 2],
    [0, 0, -1, 3],
    [-1, 1, 0, 1],
    [-1, 1, 0, 2],
    [-1, 1, 0, 3],
], dtype=float).T  # (4, 12)

_X6 = np.array([1, 0, 1, 0, 1, 0])  # index for sin components (0-indexed)
_PMW = np.array([1.0, -1.0])


def v_rotro2eu(m, r):
    """Convert rotation matrix to Euler angles.

    Parameters
    ----------
    m : str
        Rotation code string.
    r : array_like, shape (3, 3) or (3, 3, N)
        Rotation matrix/matrices.

    Returns
    -------
    e : ndarray, shape (K,) or (K, N)
        Euler angles.
    """
    mv = v_roteucode(m)
    nm = mv.shape[1] - 1  # number of rotation codes

    r = np.asarray(r, dtype=float)
    original_shape = r.shape
    squeeze = (r.ndim == 2)

    # Vectorize rotation matrices to (9, N), column-major (copy to avoid modifying input)
    if r.ndim == 2:
        rv = r.ravel(order='F').copy().reshape(9, 1)
    else:
        n = int(np.prod(original_shape[2:]))
        rv = np.zeros((9, n))
        for i in range(n):
            rv[:, i] = r[:, :, i].ravel(order='F')

    nr = rv.shape[1]

    # Transpose rotation matrix if needed
    if mv[6, -1] < 0:
        rv = rv[_RTR, :]

    ne = int(mv[1, -1])  # number of euler angles
    e = np.zeros((ne, nr))
    ef = mv[3, -1]  # scale factor

    # Process rotations in reverse order
    for i in range(nm - 1, -1, -1):
        mvi = mv[:, i]
        mi = int(mvi[0])  # rotation code (1-indexed)

        if mi <= 3:  # rotation around x, y, or z
            if mvi[5] != 0:  # skip if redundant
                # mvi[3] and mvi[4] are 1-indexed into vectorized matrix
                idx4 = int(mvi[3]) - 1  # 0-indexed
                idx5 = int(mvi[4]) - 1  # 0-indexed
                sign7 = mvi[6]

                si, ci, ri, ti = v_atan2sc(sign7 * rv[idx4, :], sign7 * rv[idx5, :])

                ei = int(mv[1, i]) - 1  # 0-indexed euler angle index
                e[ei, :] = ti * mvi[5] / ef

                # si_arr: shape (2, nr), row 0 = +mvi[5]*si, row 1 = -mvi[5]*si
                si_arr = mvi[5] * _PMW[:, np.newaxis] * si[np.newaxis, :]

                # Apply reverse rotation
                ax = mi - 1  # 0-indexed axis
                ci_rep = np.tile(ci, (6, 1))
                rv[_RTCI[:, ax], :] = ci_rep * rv[_RTCI[:, ax], :] - si_arr[_X6, :] * rv[_RTSI[:, ax], :]
        else:
            # Fixed rotation (code 4-12)
            ai = int(_SCAI[3, mi - 1]) - 1  # 0-indexed axis

            # scai values for this rotation
            cos_vals = np.tile(_SCAI[2, mi - 1], (6, 1))  # cos values
            sin_vals = _SCAI[_X6, mi - 1][:, np.newaxis]  # sin values (indexed by x6)

            rv[_RTCI[:, ai], :] = cos_vals * rv[_RTCI[:, ai], :] - sin_vals * rv[_RTSI[:, ai], :]

    if squeeze:
        return e.ravel()
    else:
        return e
