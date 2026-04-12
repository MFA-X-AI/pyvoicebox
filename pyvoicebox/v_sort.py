"""V_SORT - Sort with forward and inverse index."""

from __future__ import annotations
import numpy as np


def v_sort(a, descend=False, return_inverse=False) -> tuple[np.ndarray, np.ndarray]:
    """Sort array with forward and optional inverse index.

    Parameters
    ----------
    a : array_like
        Input vector or matrix (sorted along axis 0 for matrices).
    descend : bool, optional
        Sort in descending order.
    return_inverse : bool, optional
        If True, also return the inverse index j such that a = b[j].

    Returns
    -------
    b : ndarray
        Sorted array.
    i : ndarray
        Forward index: b = a[i].
    j : ndarray (only if return_inverse=True)
        Inverse index: a = b[j].
    """
    a = np.asarray(a)

    if a.ndim <= 1:
        if descend:
            i = np.argsort(-a)
        else:
            i = np.argsort(a)
        b = a[i]
        if return_inverse:
            j = np.empty_like(i)
            j[i] = np.arange(len(i))
            return b, i, j
        return b, i

    # Matrix case: sort each column
    if descend:
        i = np.argsort(-a, axis=0)
    else:
        i = np.argsort(a, axis=0)

    r, c = a.shape
    b = np.take_along_axis(a, i, axis=0)

    if return_inverse:
        j = np.zeros_like(i)
        for col in range(c):
            j[i[:, col], col] = np.arange(r)
        return b, i, j

    return b, i
