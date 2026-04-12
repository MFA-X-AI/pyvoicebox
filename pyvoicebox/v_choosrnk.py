"""V_CHOOSRNK - All choices of K elements from 0:N-1 with replacement."""

from __future__ import annotations
import numpy as np
from .v_choosenk import v_choosenk


def v_choosrnk(n, k) -> np.ndarray:
    """Generate all choices of K elements from 0:N-1 with replacement.

    Note: Returns 0-based indices (unlike MATLAB's 1-based).

    Parameters
    ----------
    n : int
        Range of elements (0 to n-1).
    k : int
        Number of elements to choose.

    Returns
    -------
    x : ndarray
        Each row is a combination with replacement.
    """
    x = v_choosenk(n + k - 1, k)
    x = x - np.arange(k)
    return x
