"""V_LPCLA2RF - Convert log areas to reflection coefficients."""

from __future__ import annotations
import numpy as np


def v_lpcla2rf(la) -> np.ndarray:
    """Convert log areas to reflection coefficients.

    Parameters
    ----------
    la : array_like, shape (nf, p+2)
        Log areas.

    Returns
    -------
    rf : ndarray, shape (nf, p+1)
        Reflection coefficients.
    """
    la = np.atleast_2d(np.asarray(la, dtype=float))
    nf, p2 = la.shape
    rf = -np.tanh((la[:, :p2 - 1] - la[:, 1:p2]) / 2)
    return rf
