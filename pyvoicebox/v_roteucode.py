"""V_ROTEUCODE - decode Euler angle rotation code string."""

from __future__ import annotations
import numpy as np

# Precomputed tables (from v_roteucode.m / v_rotro2eu_tab.m)
_MES = np.array([1, 2, 3, 10, 11, 12, 7, 8, 9, 4, 5, 6]) - 1  # 0-indexed sign reversal look-up

# trmap(i,j): pattern after applying rotation j (1-indexed axes j=1..12) to pattern i (1-indexed pattern)
# 52 patterns x 12 rotations (kept 1-indexed)
_TRMAP = np.array([
    [25, 29, 33, 16, 24, 11, 7, 13, 19, 22, 12, 17],
    [28, 32, 27, 17, 22, 12, 8, 14, 20, 23, 10, 18],
    [31, 26, 30, 18, 23, 10, 9, 15, 21, 24, 11, 16],
    [34, 41, 39, 13, 20, 9, 10, 16, 22, 19, 8, 15],
    [37, 35, 42, 14, 21, 7, 11, 17, 23, 20, 9, 13],
    [40, 38, 36, 15, 19, 8, 12, 18, 24, 21, 7, 14],
    [25, 38, 42, 22, 6, 23, 1, 19, 13, 16, 18, 5],
    [28, 41, 36, 23, 4, 24, 2, 20, 14, 17, 16, 6],
    [31, 35, 39, 24, 5, 22, 3, 21, 15, 18, 17, 4],
    [34, 32, 30, 19, 2, 21, 4, 22, 16, 13, 14, 3],
    [37, 26, 33, 20, 3, 19, 5, 23, 17, 14, 15, 1],
    [40, 29, 27, 21, 1, 20, 6, 24, 18, 15, 13, 2],
    [34, 29, 42, 10, 12, 5, 19, 1, 7, 4, 24, 23],
    [37, 32, 36, 11, 10, 6, 20, 2, 8, 5, 22, 24],
    [40, 26, 39, 12, 11, 4, 21, 3, 9, 6, 23, 22],
    [25, 41, 30, 7, 8, 3, 22, 4, 10, 1, 20, 21],
    [28, 35, 33, 8, 9, 1, 23, 5, 11, 2, 21, 19],
    [31, 38, 27, 9, 7, 2, 24, 6, 12, 3, 19, 20],
    [34, 38, 33, 4, 18, 17, 13, 7, 1, 10, 6, 11],
    [37, 41, 27, 5, 16, 18, 14, 8, 2, 11, 4, 12],
    [40, 35, 30, 6, 17, 16, 15, 9, 3, 12, 5, 10],
    [25, 32, 39, 1, 14, 15, 16, 10, 4, 7, 2, 9],
    [28, 26, 42, 2, 15, 13, 17, 11, 5, 8, 3, 7],
    [31, 29, 36, 3, 13, 14, 18, 12, 6, 9, 1, 8],
    [25, 44, 45, 25, 36, 26, 25, 34, 34, 25, 27, 35],
    [43, 26, 45, 27, 26, 34, 35, 26, 35, 36, 26, 25],
    [43, 44, 27, 35, 25, 27, 36, 36, 27, 26, 34, 27],
    [28, 47, 48, 28, 39, 29, 28, 37, 37, 28, 30, 38],
    [46, 29, 48, 30, 29, 37, 38, 29, 38, 39, 29, 28],
    [46, 47, 30, 38, 28, 30, 39, 39, 30, 29, 37, 30],
    [31, 50, 51, 31, 42, 32, 31, 40, 40, 31, 33, 41],
    [49, 32, 51, 33, 32, 40, 41, 32, 41, 42, 32, 31],
    [49, 50, 33, 41, 31, 33, 42, 42, 33, 32, 40, 33],
    [34, 44, 45, 34, 27, 35, 34, 25, 25, 34, 36, 26],
    [43, 35, 45, 36, 35, 25, 26, 35, 26, 27, 35, 34],
    [43, 44, 36, 26, 34, 36, 27, 27, 36, 35, 25, 36],
    [37, 47, 48, 37, 30, 38, 37, 28, 28, 37, 39, 29],
    [46, 38, 48, 39, 38, 28, 29, 38, 29, 30, 38, 37],
    [46, 47, 39, 29, 37, 39, 30, 30, 39, 38, 28, 39],
    [40, 50, 51, 40, 33, 41, 40, 31, 31, 40, 42, 32],
    [49, 41, 51, 42, 41, 31, 32, 41, 32, 33, 41, 40],
    [49, 50, 42, 32, 40, 42, 33, 33, 42, 41, 31, 42],
    [43, 52, 52, 43, 45, 44, 43, 43, 43, 43, 45, 44],
    [52, 44, 52, 45, 44, 43, 44, 44, 44, 45, 44, 43],
    [52, 52, 45, 44, 43, 45, 45, 45, 45, 44, 43, 45],
    [46, 52, 52, 46, 48, 47, 46, 46, 46, 46, 48, 47],
    [52, 47, 52, 48, 47, 46, 47, 47, 47, 48, 47, 46],
    [52, 52, 48, 47, 46, 48, 48, 48, 48, 47, 46, 48],
    [49, 52, 52, 49, 51, 50, 49, 49, 49, 49, 51, 50],
    [52, 50, 52, 51, 50, 49, 50, 50, 50, 51, 50, 49],
    [52, 52, 51, 50, 49, 51, 51, 51, 51, 50, 49, 51],
    [52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52],
], dtype=int)

# zel(k, j, i): information about which element of the rotation matrix
# changes when rotation j is applied to pattern i
# k=0: index (1-indexed) into vectorized matrix of entry that becomes non-zero
# k=1: index of other element in same column that changes
# k=2: sign of the sine term affecting entry k=0
# k=3: sign of initial value of the second entry if known
# Shape: (4, 3, 52)
_ZEL = np.array([
    [6, 5, 1, 1, 3, 1, -1, 1, 2, 1, 1, 1],
    [2, 3, -1, 1, 1, 3, 1, 1, 5, 4, 1, 1],
    [3, 2, 1, 1, 4, 6, 1, 1, 1, 2, -1, 1],
    [5, 6, -1, -1, 3, 1, -1, -1, 2, 1, 1, -1],
    [3, 2, 1, -1, 6, 4, -1, -1, 1, 2, -1, -1],
    [2, 3, -1, -1, 1, 3, 1, -1, 4, 5, -1, -1],
    [6, 5, 1, -1, 3, 1, -1, 1, 2, 1, 1, 1],
    [2, 3, -1, -1, 1, 3, 1, -1, 5, 4, 1, 1],
    [3, 2, 1, -1, 4, 6, 1, -1, 1, 2, -1, -1],
    [5, 6, -1, 1, 3, 1, -1, -1, 2, 1, 1, -1],
    [3, 2, 1, 1, 6, 4, -1, -1, 1, 2, -1, 1],
    [2, 3, -1, 1, 1, 3, 1, 1, 4, 5, -1, 1],
    [6, 5, 1, 1, 3, 1, -1, -1, 2, 1, 1, -1],
    [2, 3, -1, -1, 1, 3, 1, -1, 5, 4, 1, -1],
    [3, 2, 1, 1, 4, 6, 1, -1, 1, 2, -1, 1],
    [5, 6, -1, 1, 3, 1, -1, 1, 2, 1, 1, 1],
    [3, 2, 1, -1, 6, 4, -1, 1, 1, 2, -1, -1],
    [2, 3, -1, 1, 1, 3, 1, 1, 4, 5, -1, -1],
    [6, 5, 1, -1, 3, 1, -1, -1, 2, 1, 1, -1],
    [2, 3, -1, 1, 1, 3, 1, 1, 5, 4, 1, -1],
    [3, 2, 1, -1, 4, 6, 1, 1, 1, 2, -1, -1],
    [5, 6, -1, -1, 3, 1, -1, 1, 2, 1, 1, 1],
    [3, 2, 1, 1, 6, 4, -1, 1, 1, 2, -1, 1],
    [2, 3, -1, -1, 1, 3, 1, -1, 4, 5, -1, 1],
    [0, 0, 0, 0, 3, 1, -1, 1, 2, 1, 1, 1],
    [3, 2, 1, 1, 0, 0, 0, 0, 1, 2, -1, 1],
    [2, 3, -1, 1, 1, 3, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 6, 4, -1, 1, 5, 4, 1, 1],
    [6, 5, 1, 1, 0, 0, 0, 0, 4, 5, -1, 1],
    [5, 6, -1, 1, 4, 6, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 9, 7, -1, 1, 8, 7, 1, 1],
    [9, 8, 1, 1, 0, 0, 0, 0, 7, 8, -1, 1],
    [8, 9, -1, 1, 7, 9, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 1, -1, -1, 2, 1, 1, -1],
    [3, 2, 1, -1, 0, 0, 0, 0, 1, 2, -1, -1],
    [2, 3, -1, -1, 1, 3, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 6, 4, -1, -1, 5, 4, 1, -1],
    [6, 5, 1, -1, 0, 0, 0, 0, 4, 5, -1, -1],
    [5, 6, -1, -1, 4, 6, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 9, 7, -1, -1, 8, 7, 1, -1],
    [9, 8, 1, -1, 0, 0, 0, 0, 7, 8, -1, -1],
    [8, 9, -1, -1, 7, 9, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 3, 1, 1, 1, 2, -1, 1],
    [2, 3, -1, 1, 0, 0, 0, 0, 2, 1, 1, 1],
    [3, 2, 1, 1, 3, 1, -1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 6, 1, 1, 4, 5, -1, 1],
    [5, 6, -1, 1, 0, 0, 0, 0, 5, 4, 1, 1],
    [6, 5, 1, 1, 6, 4, -1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 9, 1, 1, 7, 8, -1, 1],
    [8, 9, -1, 1, 0, 0, 0, 0, 8, 7, 1, 1],
    [9, 8, 1, 1, 9, 7, -1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int).T.reshape(4, 3, 52, order='F')
# Note: MATLAB reads column-major, we need to handle the reshape carefully.
# The MATLAB code does: zel=reshape([...]', 4, 3, 52)
# where the data is listed row by row (each row of 12 = 4*3 elements for one i value)
# Actually let me re-derive:
# In MATLAB: zel is constructed from a 52x12 matrix (each row has 12 values = 4 values x 3 axes)
# then transposed and reshaped to (4, 3, 52)
# So for pattern i (0-indexed), the 12 values are: _ZEL_RAW[i, :] = [k1_ax1, k2_ax1, k3_ax1, k4_ax1, k1_ax2, ...]
# After transpose and reshape(4,3,52): zel(:, j, i) gives [k1, k2, k3, k4] for axis j, pattern i


def _build_zel():
    """Build the zel array from raw data matching MATLAB's reshape order."""
    raw = np.array([
        [6, 5, 1, 1, 3, 1, -1, 1, 2, 1, 1, 1],
        [2, 3, -1, 1, 1, 3, 1, 1, 5, 4, 1, 1],
        [3, 2, 1, 1, 4, 6, 1, 1, 1, 2, -1, 1],
        [5, 6, -1, -1, 3, 1, -1, -1, 2, 1, 1, -1],
        [3, 2, 1, -1, 6, 4, -1, -1, 1, 2, -1, -1],
        [2, 3, -1, -1, 1, 3, 1, -1, 4, 5, -1, -1],
        [6, 5, 1, -1, 3, 1, -1, 1, 2, 1, 1, 1],
        [2, 3, -1, -1, 1, 3, 1, -1, 5, 4, 1, 1],
        [3, 2, 1, -1, 4, 6, 1, -1, 1, 2, -1, -1],
        [5, 6, -1, 1, 3, 1, -1, -1, 2, 1, 1, -1],
        [3, 2, 1, 1, 6, 4, -1, -1, 1, 2, -1, 1],
        [2, 3, -1, 1, 1, 3, 1, 1, 4, 5, -1, 1],
        [6, 5, 1, 1, 3, 1, -1, -1, 2, 1, 1, -1],
        [2, 3, -1, -1, 1, 3, 1, -1, 5, 4, 1, -1],
        [3, 2, 1, 1, 4, 6, 1, -1, 1, 2, -1, 1],
        [5, 6, -1, 1, 3, 1, -1, 1, 2, 1, 1, 1],
        [3, 2, 1, -1, 6, 4, -1, 1, 1, 2, -1, -1],
        [2, 3, -1, 1, 1, 3, 1, 1, 4, 5, -1, -1],
        [6, 5, 1, -1, 3, 1, -1, -1, 2, 1, 1, -1],
        [2, 3, -1, 1, 1, 3, 1, 1, 5, 4, 1, -1],
        [3, 2, 1, -1, 4, 6, 1, 1, 1, 2, -1, -1],
        [5, 6, -1, -1, 3, 1, -1, 1, 2, 1, 1, 1],
        [3, 2, 1, 1, 6, 4, -1, 1, 1, 2, -1, 1],
        [2, 3, -1, -1, 1, 3, 1, -1, 4, 5, -1, 1],
        [0, 0, 0, 0, 3, 1, -1, 1, 2, 1, 1, 1],
        [3, 2, 1, 1, 0, 0, 0, 0, 1, 2, -1, 1],
        [2, 3, -1, 1, 1, 3, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 6, 4, -1, 1, 5, 4, 1, 1],
        [6, 5, 1, 1, 0, 0, 0, 0, 4, 5, -1, 1],
        [5, 6, -1, 1, 4, 6, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 9, 7, -1, 1, 8, 7, 1, 1],
        [9, 8, 1, 1, 0, 0, 0, 0, 7, 8, -1, 1],
        [8, 9, -1, 1, 7, 9, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 1, -1, -1, 2, 1, 1, -1],
        [3, 2, 1, -1, 0, 0, 0, 0, 1, 2, -1, -1],
        [2, 3, -1, -1, 1, 3, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 6, 4, -1, -1, 5, 4, 1, -1],
        [6, 5, 1, -1, 0, 0, 0, 0, 4, 5, -1, -1],
        [5, 6, -1, -1, 4, 6, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 9, 7, -1, -1, 8, 7, 1, -1],
        [9, 8, 1, -1, 0, 0, 0, 0, 7, 8, -1, -1],
        [8, 9, -1, -1, 7, 9, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 3, 1, 1, 1, 2, -1, 1],
        [2, 3, -1, 1, 0, 0, 0, 0, 2, 1, 1, 1],
        [3, 2, 1, 1, 3, 1, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 6, 1, 1, 4, 5, -1, 1],
        [5, 6, -1, 1, 0, 0, 0, 0, 5, 4, 1, 1],
        [6, 5, 1, 1, 6, 4, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 7, 9, 1, 1, 7, 8, -1, 1],
        [8, 9, -1, 1, 0, 0, 0, 0, 8, 7, 1, 1],
        [9, 8, 1, 1, 9, 7, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    # raw is (52, 12). In MATLAB: zel = reshape(raw', 4, 3, 52)
    # raw' is (12, 52), then reshape column-major to (4, 3, 52)
    zel = raw.T.reshape(4, 3, 52, order='F')
    return zel


ZEL = _build_zel()

# mes: sign reversal look-up (1-indexed values; maps code 1..12 to reversed code)
MES = np.array([1, 2, 3, 10, 11, 12, 7, 8, 9, 4, 5, 6], dtype=int)


def v_roteucode(m) -> np.ndarray:
    """Decode a string specifying a rotation axis sequence.

    Parameters
    ----------
    m : str
        Rotation code string.

    Returns
    -------
    mv : ndarray, shape (7, k)
        Decoded rotation parameters where k-1 is the number of rotations.
        mv[0, j] = rotation code (1-indexed: 1,2,3 for x,y,z; 4-12 for fixed)
        mv[1, j] = index into euler angle array (1-indexed)
        mv[2, j] = rotation class (pattern ID, 1-indexed)
        mv[3, j] = index into vectorized matrix (1-indexed)
        mv[4, j] = index of other changing element
        mv[5, j] = sign of sine term
        mv[6, j] = sign of entry before rotation / inversion flag (last col)
        mv[3, -1] = scale factor for euler angles
        mv[6, -1] = -1 to invert rotation (transpose matrix) or +1
    """
    # Convert characters to integers (m - 'w')
    mm_raw = np.array([ord(c) - ord('w') for c in m], dtype=int)

    # Find characters XYZ (uppercase) and convert to xyz
    mi_upper = (mm_raw >= -31) & (mm_raw <= -29)
    mm_raw[mi_upper] += 32

    # Find digits 1-9 and convert to codes 4-12
    mi_digit = (mm_raw >= -70) & (mm_raw <= -62)
    mm_raw[mi_digit] += 74

    # Separate control characters (<=0) from rotations (>0)
    mi_ctrl = mm_raw <= 0
    mc = mm_raw[mi_ctrl]
    mm = mm_raw[~mi_ctrl]

    # Process control characters
    ef = 1.0  # angle scale factor
    es = 1    # angle sign
    fl = 1    # no rotation matrix transposing

    for c in mc:
        if c == -5:     # 'r' = radians
            pass
        elif c == -19:  # 'd' = degrees
            ef = np.pi / 180.0
        elif c == -37:  # 'R' = negated radians
            ef = -1.0
        elif c == -51:  # 'D' = negated degrees
            ef = -np.pi / 180.0
        elif c == -8:   # 'o' = object-extrinsic
            pass
        elif c == -40:  # 'O' = object-intrinsic
            fl = -1
            es = -1
        elif c == -22:  # 'a' = axes-extrinsic
            fl = -1
        elif c == -54:  # 'A' = axes-intrinsic
            es = -1
        else:
            raise ValueError(f'Invalid character: {chr(c + ord("w"))}')

    ef = ef * es  # change sign of scale factor if necessary
    if es < 0:
        mm = MES[mm - 1]  # sign-reverse: interchange 4,5,6 with 10,11,12

    nm = len(mm)
    mv = np.zeros((7, nm + 1), dtype=float)
    mv[0, :nm] = mm
    mv[0, nm] = 0

    # Cumulative sum for euler angle indices
    euler_mask = np.concatenate([(mm <= 3).astype(int), [0]])
    mv[1, :] = np.cumsum(euler_mask)

    mv[2, 0] = 1  # initial pattern is identity matrix (pattern 1)
    for i in range(nm):
        mmi = int(mm[i])  # rotation code (1-indexed)
        mv[2, i + 1] = _TRMAP[int(mv[2, i]) - 1, mmi - 1]  # pattern ID after rotation
        if mmi <= 3:
            # zel is (4, 3, 52) with 1-indexed pattern in dim 2
            mv[3:7, i] = ZEL[:, mmi - 1, int(mv[2, i]) - 1]

    mv[6, nm] = fl
    mv[3, nm] = ef

    return mv
