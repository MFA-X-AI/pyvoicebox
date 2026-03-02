"""V_CHOOSENK - All choices of K elements from 0:N-1."""

import numpy as np
from itertools import combinations


def v_choosenk(n, k):
    """Generate all choices of K elements from 0:N-1 in lexical order.

    Note: Returns 0-based indices (unlike MATLAB's 1-based).

    Parameters
    ----------
    n : int
        Range of elements (0 to n-1).
    k : int
        Number of elements to choose.

    Returns
    -------
    x : ndarray of shape (C(n,k), k)
        Each row is a combination.
    """
    if k > n:
        return np.array([]).reshape(0, max(k, 0))
    if k == 0:
        return np.array([]).reshape(1, 0)

    result = list(combinations(range(n), k))
    x = np.array(result, dtype=int)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x
