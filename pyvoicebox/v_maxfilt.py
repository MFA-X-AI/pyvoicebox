"""V_MAXFILT - Find max of an exponentially weighted sliding window."""

import numpy as np


def v_maxfilt(x, f=1, n=None, d=None, x0=None):
    """Find max of an exponentially weighted sliding window.

    Calculates y(p) = max(f^r * x(p-r), r=0:n-1) where x(r)=-inf for r<0.

    Parameters
    ----------
    x : array_like
        Input data (vector or matrix).
    f : float, optional
        Exponential forgetting factor, range [0, 1]. Default 1 (no forgetting).
    n : int or None, optional
        Length of sliding window. Default is inf.
    d : int or None, optional
        Dimension to work along (0-based). Default is first non-singleton.
    x0 : array_like or None, optional
        Initial values placed in front of x data.

    Returns
    -------
    y : ndarray
        Output array, same shape as x.
    k : ndarray
        Index array such that y = x[k] (0-based).
    y0 : ndarray
        Last n-1 values for subsequent calls (or last output if n=inf).
    """
    x = np.asarray(x, dtype=float)
    s = x.shape

    if d is None:
        d_found = None
        for i, si in enumerate(s):
            if si > 1:
                d_found = i
                break
        d = d_found if d_found is not None else 0

    # Concatenate x0 if provided
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float)
        if x0.size > 0:
            y = np.concatenate([x0, x], axis=d)
            nx0 = x0.shape[d]
        else:
            y = x.copy()
            nx0 = 0
    else:
        y = x.copy()
        nx0 = 0

    # Move working dimension to axis 0
    y = np.moveaxis(y, d, 0)
    s_y = y.shape
    s1 = s_y[0]

    if n is None:
        n0 = np.inf
    else:
        n0 = max(n, 1)

    nn = n0
    nn = int(min(nn, s1)) if not np.isinf(nn) else s1

    # Reshape to 2D: (s1, rest)
    rest_shape = s_y[1:]
    rest_size = int(np.prod(rest_shape)) if len(rest_shape) > 0 else 1
    y_2d = y.reshape(s1, rest_size)
    k_2d = np.tile(np.arange(s1)[:, np.newaxis], (1, rest_size))

    # The MATLAB algorithm: doubling step max-propagation
    if nn > 1:
        j = 1
        j2 = 1
        while j > 0:
            g = f ** j
            # Compare y[j:] with g*y[:-j]
            for col in range(rest_size):
                mask = y_2d[j:, col] <= g * y_2d[:s1 - j, col]
                rows = np.where(mask)[0]
                y_2d[rows + j, col] = g * y_2d[rows, col]
                k_2d[rows + j, col] = k_2d[rows, col]
            j2 = j2 + j
            nj = nn - j2
            if nj <= 0:
                j = 0
            else:
                j = min(j2, nj)

    # Compute y0 output
    if not np.isinf(n0):
        ny0 = min(s1, nn - 1)
    else:
        ny0 = min(s1, 1)

    if ny0 <= 0 or np.isinf(n0):
        # For n=inf, y0 is the last output
        if np.isinf(n0) and ny0 == 1:
            y0_2d = y_2d[-1:, :]
        else:
            y0_2d = np.zeros((max(ny0, 0), rest_size))
    else:
        y0_2d = y_2d[s1 - ny0:, :]

    # Remove prepended x0 data
    if nx0 > 0:
        y_2d = y_2d[nx0:, :]
        k_2d = k_2d[nx0:, :] - nx0
        s1_out = s1 - nx0
    else:
        s1_out = s1

    # Reshape back
    out_shape = (s1_out,) + rest_shape
    y_out = y_2d.reshape(out_shape)
    k_out = k_2d.reshape(out_shape)

    # Move axis back
    y_out = np.moveaxis(y_out, 0, d)
    k_out = np.moveaxis(k_out, 0, d)

    # Reshape y0
    y0_shape = list(s)
    y0_shape[d] = ny0
    y0_2d_reshaped = y0_2d.reshape((ny0,) + rest_shape)
    y0_out = np.moveaxis(y0_2d_reshaped, 0, d)

    return y_out, k_out, y0_out
