"""V_SKEW3D - convert between vector and skew-symmetric matrix."""

import numpy as np


def v_skew3d(x, m=''):
    """Convert between a vector and the corresponding skew-symmetric matrix.

    v_skew3d is its own inverse: v_skew3d(v_skew3d(x)) == x.

    Parameters
    ----------
    x : array_like
        Input vector or matrix. Size must be (3,), (3, 3), (6,), or (4, 4).
    m : str, optional
        Mode string: 'n' normalize, 'z' orthogonalize.

    Returns
    -------
    y : ndarray
        Output matrix or vector.
    """
    x = np.asarray(x, dtype=float)
    mn = 'n' in m
    mz = 'z' in m

    j, k = x.shape if x.ndim == 2 else (x.shape[0], 1)

    if j == 3 and k == 1:
        # 3x1 vector -> 3x3 skew-symmetric matrix
        xv = x.ravel()
        if mn and xv @ xv > 0:
            xv = xv / np.sqrt(xv @ xv)
        y = np.zeros((3, 3))
        # MATLAB column-major flat indices: [6,7,2] -> (2,0), (0,1), (2,0) in row-major
        # Actually MATLAB flat indices (1-based, column-major): 6->row2,col1; 7->row0,col2; 2->row1,col0
        # 0-based column-major: index 5 -> (2,1), index 6 -> (0,2), index 1 -> (1,0)
        y[2, 1] = xv[0]
        y[0, 2] = xv[1]
        y[1, 0] = xv[2]
        y[1, 2] = -xv[0]
        y[2, 0] = -xv[1]
        y[0, 1] = -xv[2]
        return y
    elif j == 3 and k == 3:
        # 3x3 skew-symmetric matrix -> 3x1 vector
        # Extract from positions [6,7,2] (MATLAB 1-based column-major) = (2,1), (0,2), (1,0)
        y = np.array([x[2, 1], x[0, 2], x[1, 0]])
        if mn and y @ y > 0:
            y = y / np.sqrt(y @ y)
        return y
    elif j == 6 and k == 1:
        # 6x1 Plucker vector -> 4x4 Plucker matrix
        xv = x.ravel().copy()
        u = xv[:3]
        v = xv[5:2:-1]  # [x[5], x[4], x[3]]
        if mz and u @ u > 0 and v @ v > 0:
            v = v - (u @ v) / (2 * (u @ u)) * u
            xv = np.concatenate([u - (v @ u) / (v @ v) * v, v[::-1]])
        if mn and xv @ xv > 0:
            xv = xv / np.sqrt(xv @ xv)
        y = np.zeros((4, 4))
        # MATLAB 1-based column-major indices for positive entries: [5,9,13,10,8,15]
        # 0-based column-major: [4,8,12,9,7,14]
        # -> (0,1), (0,2), (0,3), (1,2), (3,1), (2,3)
        y[0, 1] = xv[0]
        y[0, 2] = xv[1]
        y[0, 3] = xv[2]
        y[1, 2] = xv[3]
        y[3, 1] = xv[4]
        y[2, 3] = xv[5]
        # Negative entries: [2,3,4,7,14,12] -> 0-based col-major: [1,2,3,6,13,11]
        # -> (1,0), (2,0), (3,0), (2,1), (1,3), (3,2)
        y[1, 0] = -xv[0]
        y[2, 0] = -xv[1]
        y[3, 0] = -xv[2]
        y[2, 1] = -xv[3]
        y[1, 3] = -xv[4]
        y[3, 2] = -xv[5]
        return y
    elif j == 4 and k == 4:
        # 4x4 Plucker matrix -> 6x1 Plucker vector
        # MATLAB 1-based column-major: u = x([5,9,13]) = (0,1), (0,2), (0,3)
        u = np.array([x[0, 1], x[0, 2], x[0, 3]])
        # v = x([15,8,10]) = (2,3), (3,1), (1,2)
        v = np.array([x[2, 3], x[3, 1], x[1, 2]])
        if mz and u @ u > 0 and v @ v > 0:
            v = v - (u @ v) / (2 * (u @ u)) * u
            y = np.concatenate([u - (v @ u) / (v @ v) * v, v[::-1]])
        else:
            y = np.concatenate([u, v[::-1]])
        if mn and y @ y > 0:
            y = y / np.sqrt(y @ y)
        return y
    else:
        raise ValueError('size(x) must be (3,), (3, 3), (6,), or (4, 4)')
