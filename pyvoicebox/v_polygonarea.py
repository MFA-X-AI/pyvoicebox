"""V_POLYGONAREA - calculate polygon area."""

import numpy as np


def v_polygonarea(p):
    """Calculate the area of a polygon using the shoelace formula.

    Parameters
    ----------
    p : array_like, shape (n, 2)
        Polygon vertices.

    Returns
    -------
    a : float
        Signed area (positive if vertices are counter-clockwise).
    """
    p = np.asarray(p, dtype=float)
    # Append first point
    p_closed = np.vstack([p, p[0:1, :]])
    a = 0.5 * np.sum(
        (p_closed[:-1, 0] - p_closed[1:, 0]) * (p_closed[:-1, 1] + p_closed[1:, 1])
    )
    return float(a)
