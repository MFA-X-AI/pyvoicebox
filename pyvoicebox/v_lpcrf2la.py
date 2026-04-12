"""V_LPCRF2LA - Convert reflection coefficients to log areas."""

from __future__ import annotations
import numpy as np


def v_lpcrf2la(rf) -> np.ndarray:
    """Convert reflection coefficients to log areas.

    Parameters
    ----------
    rf : array_like, shape (nf, p+1)
        Reflection coefficients.

    Returns
    -------
    la : ndarray, shape (nf, p+2)
        Log areas, normalized so la[:, -1] = 0.
    """
    rf = np.atleast_2d(np.asarray(rf, dtype=float))
    nf, p1 = rf.shape
    r = np.clip(rf, -1 + 1e-6, 1 - 1e-6)
    lo = np.log((1 - r) / (1 + r))
    cs = np.cumsum(lo[:, ::-1], axis=1).T  # cumsum along flipped columns
    la = np.column_stack([np.zeros((nf, 1)), cs.T])[:, ::-1]
    return la
