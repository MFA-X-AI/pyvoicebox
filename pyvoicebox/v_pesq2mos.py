"""V_PESQ2MOS - Convert PESQ speech quality scores to MOS."""

from __future__ import annotations
import numpy as np


def v_pesq2mos(p) -> np.ndarray:
    """Convert PESQ speech quality scores to MOS.

    Parameters
    ----------
    p : array_like
        Matrix of PESQ scores.

    Returns
    -------
    m : ndarray
        Matrix of MOS scores, same shape as p.

    References
    ----------
    [1] ITU-T Recommendation P.862.1, Nov. 2003.
    """
    p = np.asarray(p, dtype=float)
    a = 0.999
    b = 4.999 - a
    c = -1.4945
    d = 4.6607
    m = a + b / (1.0 + np.exp(c * p + d))
    return m
