"""V_ZEROTRIM - Remove trailing zero rows and columns."""

from __future__ import annotations
import numpy as np


def v_zerotrim(x) -> np.ndarray:
    """Remove trailing zero rows and columns from a matrix.

    Parameters
    ----------
    x : array_like
        Input matrix.

    Returns
    -------
    z : ndarray
        Trimmed matrix, or empty array if all zeros.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        nz = np.nonzero(x)[0]
        if len(nz) == 0:
            return np.array([])
        return x[:nz[-1] + 1]

    # Find last nonzero column
    col_any = np.any(x, axis=0)
    col_nz = np.nonzero(col_any)[0]
    if len(col_nz) == 0:
        return np.array([]).reshape(0, 0)
    c = col_nz[-1] + 1

    # Find last nonzero row
    row_any = np.any(x, axis=1)
    row_nz = np.nonzero(row_any)[0]
    r = row_nz[-1] + 1

    return x[:r, :c]
