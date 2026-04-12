"""V_MOS2PESQ - Convert MOS speech quality scores to PESQ."""

from __future__ import annotations
import numpy as np


def v_mos2pesq(m) -> np.ndarray:
    """Convert MOS speech quality scores to PESQ.

    Parameters
    ----------
    m : array_like
        Matrix of MOS scores.

    Returns
    -------
    p : ndarray
        Matrix of PESQ scores, same shape as m.

    References
    ----------
    [1] ITU-T Recommendation P.862.1, Nov. 2003.
    """
    m = np.asarray(m, dtype=float)
    a = 0.999
    b = 4.999 - a
    c = -1.4945
    d = 4.6607
    p = (np.log(b / (m - a) - 1.0) - d) / c
    return p
