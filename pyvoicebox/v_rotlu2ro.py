"""V_ROTLU2RO - convert look and up directions to rotation matrix."""

import numpy as np


def v_rotlu2ro(l, u=None):
    """Convert look and up directions to a rotation matrix.

    The rotation maps the look direction to the negative z-axis and the
    up direction to lie in the y-z plane with a positive y component.

    Parameters
    ----------
    l : array_like, shape (3,) or (3, N)
        Look direction vector(s).
    u : array_like, shape (3,) or (3, N), optional
        Up direction vector(s). Default is [0, 0, 1] unless l is a
        multiple of this, in which case [0, 1, 0].

    Returns
    -------
    r : ndarray, shape (3, 3) or (3, 3, N)
        Rotation matrix/matrices.
    """
    l = np.asarray(l, dtype=float)
    if l.ndim == 1:
        l = l.reshape(3, 1)
        squeeze = True
    else:
        squeeze = False

    n = l.shape[1]

    # mk: sign correction masks, shape (3, 3, 4)
    mk = np.zeros((3, 3, 4))
    mk[:, :, 0] = np.tile(np.array([-1, 1, -1]).reshape(3, 1), (1, 3))
    mk[:, :, 1] = np.tile(np.array([1, -1, -1]).reshape(3, 1), (1, 3))
    mk[:, :, 2] = np.tile(np.array([1, 1, 1]).reshape(3, 1), (1, 3))
    mk[:, :, 3] = np.tile(np.array([-1, -1, 1]).reshape(3, 1), (1, 3))

    if u is None:
        u = np.tile(np.array([0, 0, 1], dtype=float).reshape(3, 1), (1, n))
        # If l is along z-axis, use [0, 1, 0] instead
        along_z = (l[0, :] == 0) & (l[1, :] == 0)
        u[1, along_z] += 1.0

    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u.reshape(3, 1)

    if squeeze:
        r = np.zeros((3, 3))
    else:
        r = np.zeros((3, 3, n))

    for i in range(n):
        li = l[:, i:i + 1]
        ui = u[:, i:i + 1]
        # QR decomposition
        q_mat, t_mat = np.linalg.qr(np.hstack([li, ui]))

        # rx = [cross(q2, q1), q2, q1]^T
        q1 = q_mat[:, 0]
        q2 = q_mat[:, 1]
        rx = np.vstack([np.cross(q2, q1), q2, q1])

        # Select sign correction based on orientation
        cond1 = int(rx[2, :] @ li[:, 0] < 0)  # rx(3,:)*l < 0
        cond2 = int(rx[1, :] @ ui[:, 0] < 0)  # rx(2,:)*u < 0
        mk_idx = 2 * cond1 + cond2

        ri = rx * mk[:, :, mk_idx]
        if squeeze:
            r = ri
        else:
            r[:, :, i] = ri

    return r
