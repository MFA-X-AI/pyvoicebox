"""V_POLYGONWIND - test if points are inside a polygon."""

import numpy as np


def v_polygonwind(p, x):
    """Calculate the winding number for points with respect to a polygon.

    Parameters
    ----------
    p : array_like, shape (n, 2)
        Polygon vertices.
    x : array_like, shape (m, 2)
        Points to test.

    Returns
    -------
    w : ndarray, shape (m,)
        Winding number for each point. For a CCW polygon, 0=outside, 1=inside.
    """
    p = np.asarray(p, dtype=float)
    x = np.asarray(x, dtype=float)
    n = p.shape[0]
    m = x.shape[0]

    # Close the polygon
    q = np.zeros((2, n + 1))
    q[:, :n] = p.T
    q[:, n] = q[:, 0]

    # Indices for edges: i -> j
    ii = np.arange(n)
    jj = np.arange(1, n + 1)

    # Cross product term for each (point, edge) pair
    # q(1,i)*q(2,j) - q(2,i)*q(1,j) + x(:,1)*(q(2,i)-q(2,j)) + x(:,2)*(q(1,j)-q(1,i))
    cross_const = q[0, ii] * q[1, jj] - q[1, ii] * q[0, jj]  # (n,)
    dy = q[1, ii] - q[1, jj]  # (n,)
    dx = q[0, jj] - q[0, ii]  # (n,)

    # cross_val(m, n)
    cross_val = (cross_const[np.newaxis, :]
                 + x[:, 0:1] * dy[np.newaxis, :]
                 + x[:, 1:2] * dx[np.newaxis, :])

    sign_term = 2 * (cross_val > 0).astype(float) - 1  # (m, n)

    # Edge crossing test: y coordinates
    above_i = (q[1, ii][np.newaxis, :] > x[:, 1:2])  # (m, n)
    above_j = (q[1, jj][np.newaxis, :] > x[:, 1:2])  # (m, n)

    crossing = np.abs(above_j.astype(float) - above_i.astype(float))  # (m, n)

    w = np.sum(sign_term * crossing, axis=1) / 2.0
    return w.astype(int) if np.all(w == np.round(w)) else w
