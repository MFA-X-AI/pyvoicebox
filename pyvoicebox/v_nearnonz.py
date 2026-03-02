"""V_NEARNONZ - Replace each zero element with nearest non-zero element."""

import numpy as np
from scipy import sparse


def v_nearnonz(x, d=None):
    """Replace each zero element with the nearest non-zero element.

    Parameters
    ----------
    x : array_like
        Input vector, matrix or larger array.
    d : int or None, optional
        Dimension to apply filter along (0-based).
        Default is first non-singleton dimension.

    Returns
    -------
    v : ndarray
        Same shape as x, with zeros replaced by nearest non-zero value.
    y : ndarray
        Same shape as x, index along dimension d from which v was taken (1-based).
    w : ndarray
        Same shape as x, distance to the nearest non-zero entry.
    """
    x = np.asarray(x, dtype=float)
    e = x.shape
    p = x.size

    if d is None:
        d_found = None
        for i, si in enumerate(e):
            if si > 1:
                d_found = i
                break
        d = d_found if d_found is not None else 0

    k = e[d]  # size of active dimension
    q = p // k  # size of remainder

    # Move working dimension to front and reshape to 2D
    z = np.moveaxis(x, d, 0)
    r = z.shape
    z = z.reshape(k, q)

    xx = (z != 0)  # boolean array of non-zero positions
    cx = np.cumsum(xx, axis=0)  # cumulative count of non-zeros

    v_out = np.zeros((k, q))
    y_out = np.zeros((k, q), dtype=int)
    w_out = np.zeros((k, q), dtype=int)

    for col in range(q):
        nz_indices = np.where(xx[:, col])[0]
        if len(nz_indices) == 0:
            # No non-zero entries: y=0, v=0, w=0
            continue

        # Build the position array: for each element find the nearest non-zero
        for row in range(k):
            # Find nearest non-zero
            dists = np.abs(nz_indices - row)
            # Tie-break: higher index wins (>=)
            min_dist = np.min(dists)
            candidates = nz_indices[dists == min_dist]
            nearest = candidates[-1]  # take higher index for tie-break
            y_out[row, col] = nearest + 1  # 1-based
            v_out[row, col] = z[nearest, col]
            w_out[row, col] = nearest - row

    # Reshape back
    v_out = v_out.reshape(r)
    y_out = y_out.reshape(r)
    w_out = w_out.reshape(r)

    v_out = np.moveaxis(v_out, 0, d)
    y_out = np.moveaxis(y_out, 0, d)
    w_out = np.moveaxis(w_out, 0, d)

    return v_out, y_out, w_out
