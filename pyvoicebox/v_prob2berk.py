"""V_PROB2BERK - Convert probability to Berksons (log-odds base 2)."""

from __future__ import annotations
import numpy as np


def v_prob2berk(p) -> tuple[np.ndarray, np.ndarray]:
    """Convert probability to Berksons.

    Parameters
    ----------
    p : array_like
        Probability values.

    Returns
    -------
    b : ndarray
        Corresponding Berkson values.
    d : ndarray
        Corresponding derivatives dP/dB.
    """
    p = np.asarray(p, dtype=float)
    b = np.log2(p / (1.0 - p))
    d = np.log(2.0) * p * (1.0 - p)
    return b, d
