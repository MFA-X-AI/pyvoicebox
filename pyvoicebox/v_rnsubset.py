"""V_RNSUBSET - Choose k distinct random integers from 1:n."""

import numpy as np


def v_rnsubset(k, n):
    """Choose k distinct random integers from 0:n-1.

    Note: Returns 0-based indices (unlike MATLAB's 1-based).

    Parameters
    ----------
    k : int
        Number of distinct integers required.
    n : int
        Range is 0 to n-1.

    Returns
    -------
    m : ndarray
        Array of k distinct random integers.
    """
    if k > n:
        raise ValueError('k must be <= n')

    f, e = np.frexp(n)
    if k > 0.03 * n * (e - 1):
        # For large k, random permutation
        m = np.random.permutation(n)
    else:
        # Fisher-Yates partial shuffle
        m = np.arange(n)
        for i in range(k):
            j = i + np.random.randint(n - i)
            m[i], m[j] = m[j], m[i]

    return m[:k]
