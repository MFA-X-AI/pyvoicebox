"""V_PERMUTES - All N! permutations of 0:N-1 + signatures."""

from __future__ import annotations
import numpy as np


def v_permutes(n, return_sign=False) -> np.ndarray:
    """Generate all N! permutations of 0:N-1 in lexical order.

    Note: Returns 0-based indices (unlike MATLAB's 1-based).

    Parameters
    ----------
    n : int
        Number of elements to permute.
    return_sign : bool, optional
        If True, also return the signature of each permutation.

    Returns
    -------
    p : ndarray of shape (n!, n)
        Each row is a permutation.
    s : ndarray of shape (n!,), optional
        Signature (+1 or -1) of each permutation.
    """
    p = np.array([[0]])
    m = 1
    if n > 1:
        for a in range(1, n):
            nrows = (a + 1) * m
            q = np.zeros((nrows, a + 1), dtype=int)
            r = np.arange(1, a + 2)  # [1, 2, ..., a+1] then map to 0-based
            ix = np.arange(m)
            for b in range(a + 1):
                q[ix, 0] = b
                # Map p values through r (adjusting indices)
                q[ix, 1:a + 1] = r[p]
                r[b] = b
                ix = ix + m
            m = m * (a + 1)
            p = q

    if return_sign:
        s = 1 - 2 * (np.arange(1, m + 1) // 2 % 2)
        return p, s
    return p
