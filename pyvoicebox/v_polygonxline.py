"""V_POLYGONXLINE - find where a line crosses a polygon."""

from __future__ import annotations
import numpy as np


def v_polygonxline(p, l) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find where a line crosses a polygon.

    Parameters
    ----------
    p : array_like, shape (n, 2)
        Polygon vertices.
    l : array_like, shape (3,)
        Line in the form l @ [x, y, 1] = 0.

    Returns
    -------
    xc : ndarray, shape (k, 2)
        Crossing point coordinates.
    ec : ndarray, shape (k,)
        Edge indices (1-indexed) where crossings occur.
    tc : ndarray, shape (k,)
        Parametric positions along the line.
    xy0 : ndarray, shape (2,)
        Starting point of the parametric line.
    """
    p = np.asarray(p, dtype=float)
    l = np.asarray(l, dtype=float).ravel()
    n = p.shape[0]

    # Close polygon
    q = np.ones((n + 1, 3))
    q[:n, :2] = p
    q[n, :] = q[0, :]

    # Signed distance of each vertex from the line
    cdist = q @ l  # (n+1,)
    cside = cdist > 0
    cside[n] = cside[0]

    # Find edges that cross the line (where side changes)
    ec = np.where(cside[1:n + 1] != cside[:n])[0]  # 0-indexed edge indices
    nc = len(ec)

    if nc == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int), np.empty(0), np.empty(0)

    l2 = l[:2]
    l2m = l2 @ l2
    l3 = l[2]
    a = np.array([-l[1], l[0]])
    xy0 = -l3 / l2m * l2

    # Parametric position along the line
    tn = (q[:, :2] - xy0[np.newaxis, :]) @ a / l2m  # (n+1,)
    tc = (cdist[ec + 1] * tn[ec] - cdist[ec] * tn[ec + 1]) / (cdist[ec + 1] - cdist[ec])

    # Sort crossings
    idx = np.argsort(tc)
    tc = tc[idx]
    ec = ec[idx]

    # Remove duplicate pairs
    if len(tc) > 1:
        tm = tc[1:] == tc[:-1]
        tm_full = np.concatenate([[False], tm]) | np.concatenate([tm, [False]])
        tc = tc[~tm_full]
        ec = ec[~tm_full]

    nc = len(ec)
    xc = xy0[np.newaxis, :] + tc[:, np.newaxis] * a[np.newaxis, :]

    # Convert to 1-indexed edge indices (MATLAB convention)
    ec = ec + 1

    return xc, ec, tc, xy0
