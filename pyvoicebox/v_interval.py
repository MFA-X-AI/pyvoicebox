"""V_INTERVAL - Classify X values into contiguous intervals."""

from __future__ import annotations
import numpy as np
from pyvoicebox.v_sort import v_sort


def v_interval(x, y, m='') -> tuple[np.ndarray, np.ndarray]:
    """Classify X values into contiguous intervals with boundaries from Y.

    Parameters
    ----------
    x : array_like
        Vector of test values.
    y : array_like
        Vector of monotonically increasing interval boundaries.
        Interval i is [y[i], y[i+1]).
    m : str, optional
        Mode options:
          For x < y[0]:
            'e' - extrapolate: i=1, f<0 (default)
            'c' - clip: i=1, f=0
            'n' - NaN: i=NaN, f=NaN
            'z' - zero: i=0, f<0
          For x >= y[-1]:
            'E' - extrapolate: i=ny-1, f>1 (default)
            'C' - clip: i=ny-1, f=1
            'N' - NaN: i=NaN, f=NaN
            'Z' - zero: i=ny, f>1

    Returns
    -------
    i : ndarray
        Interval indices (1-based, matching MATLAB convention).
    f : ndarray
        Fractional position within the interval.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    x_shape = x.shape
    x_flat = x.ravel()
    ny = len(y)

    # Use searchsorted to find where each x would be inserted in y
    # This gives the index of the first element in y that is > x
    # k = number of y values <= x, equivalent to MATLAB's logic
    k = np.searchsorted(y, x_flat, side='right')  # k in range [0, ny]

    # Force i to lie in range [1, ny-1] (1-based)
    i = np.clip(k, 1, ny - 1)

    # Fractional position: f = (x - y[i-1]) / (y[i] - y[i-1])
    # i is 1-based, so y[i-1] and y[i] in 0-based indexing
    f = (x_flat - y[i - 1]) / (y[i] - y[i - 1])

    # Handle values below range
    klo = k < 1
    if np.any(klo):
        if 'c' in m:
            f[klo] = 0
        elif 'n' in m:
            i = i.astype(float)
            i[klo] = np.nan
            f[klo] = np.nan
        elif 'z' in m:
            i = i.astype(float)
            i[klo] = 0

    # Handle values above range
    khi = k >= ny
    if np.any(khi):
        if 'C' in m:
            f[khi] = 1
        elif 'N' in m:
            if not np.issubdtype(i.dtype, np.floating):
                i = i.astype(float)
            i[khi] = np.nan
            f[khi] = np.nan
        elif 'Z' in m:
            if not np.issubdtype(i.dtype, np.floating):
                i = i.astype(float)
            i[khi] = ny

    i = np.reshape(i, x_shape)
    f = np.reshape(f, x_shape)

    return i, f
