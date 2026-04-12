"""V_ROTATION - encode and decode rotation matrices."""

from __future__ import annotations
import numpy as np


def v_rotation(x, y=None, z=None) -> np.ndarray:
    """Create or decompose a rotation matrix.

    (1) r = v_rotation(x, y, angle): rotation in the plane of x and y
    (2) axis, angle = v_rotation(r): decompose 3x3 rotation matrix
    (3) r = v_rotation(axis, angle): rotation about axis (3D only)
    (4) r = v_rotation(axis_times_angle): rotation about axis (3D only)

    Parameters
    ----------
    x : array_like
        First vector, rotation matrix, or axis*angle vector.
    y : array_like or float, optional
        Second vector or angle.
    z : float, optional
        Rotation angle.

    Returns
    -------
    Depends on usage mode; see above.
    """
    x = np.asarray(x, dtype=float)

    if z is not None:
        # Mode (1): r = v_rotation(x, y, angle)
        x_vec = x.ravel()
        l = len(x_vec)
        x_vec = x_vec / np.sqrt(x_vec @ x_vec)
        y_vec = np.asarray(y, dtype=float).ravel()
        y_vec = y_vec - (y_vec @ x_vec) * x_vec
        y_vec = y_vec / np.sqrt(y_vec @ y_vec)
        angle = float(z)
        r = (np.eye(l) + (np.cos(angle) - 1) * (np.outer(x_vec, x_vec) + np.outer(y_vec, y_vec))
             + np.sin(angle) * (np.outer(y_vec, x_vec) - np.outer(x_vec, y_vec)))
        return r

    elif x.ndim == 1 or (x.ndim == 2 and min(x.shape) == 1):
        # x is a vector
        x_vec = x.ravel()
        l = len(x_vec)
        if l == 3:
            # Mode (3) or (4): rotation about axis
            lx = np.sqrt(x_vec @ x_vec)
            if y is None:
                angle = lx
            else:
                angle = float(y)
            x_vec = x_vec / lx
            xx = np.outer(x_vec, x_vec)
            # Build skew-symmetric matrix
            # MATLAB: s([6 7 2])=x means s(row2,col1)=x(0), s(row0,col2)=x(1), s(row1,col0)=x(2)
            # in 0-based column-major flat indexing: 5->r(2,1), 6->r(0,2), 1->r(1,0)
            s = np.zeros((3, 3))
            s[2, 1] = x_vec[0]
            s[0, 2] = x_vec[1]
            s[1, 0] = x_vec[2]
            s[1, 2] = -x_vec[0]
            s[2, 0] = -x_vec[1]
            s[0, 1] = -x_vec[2]
            r = xx + np.cos(angle) * (np.eye(3) - xx) + np.sin(angle) * s
            return r
        else:
            raise ValueError("For non-3D vectors, use the 3-argument form")
    else:
        # x is a matrix: decomposition mode
        r_mat = x
        n = r_mat.shape[0]

        # Use eigendecomposition
        from scipy.linalg import schur
        # scipy.linalg.schur returns (T, Z) where A = Z @ T @ Z^H
        e, q = schur(r_mat, output='complex')
        d = np.diag(e)
        j = np.argsort(np.real(d))
        j1 = j[0]
        ang = np.angle(d[j1])

        sq = np.sqrt(2)
        r_vec = np.imag(q[:, j1]) * sq
        if np.all(np.abs(r_vec) < 1e-10):
            p_vec = q[:, j1].real
            r_vec = q[:, j[1]].real
        else:
            p_vec = np.real(q[:, j1]) * sq

        if n == 3:
            axis = np.cross(r_vec, p_vec)
            return axis, ang
        else:
            return r_vec, p_vec, ang
