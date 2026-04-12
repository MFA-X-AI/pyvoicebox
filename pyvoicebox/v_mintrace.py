"""V_MINTRACE - Find row permutation to minimize trace."""

from __future__ import annotations
import numpy as np
from pyvoicebox.v_permutes import v_permutes


def v_mintrace(x) -> np.ndarray:
    """Find row permutation to minimize trace of x(p,:).

    Parameters
    ----------
    x : array_like, shape (n, n)
        Square real-valued matrix.

    Returns
    -------
    p : ndarray, shape (n,)
        Row permutation that minimizes trace(x[p, :]).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    perms = v_permutes(n)  # shape (n!, n)
    # Compute trace for each permutation
    cols = np.arange(n)
    traces = np.sum(x[perms, cols], axis=1)
    best = np.argmin(traces)
    return perms[best, :]
