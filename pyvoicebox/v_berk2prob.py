"""V_BERK2PROB - Convert Berksons (log-odds base 2) to probability."""

from __future__ import annotations
import numpy as np


def v_berk2prob(b) -> tuple[np.ndarray, np.ndarray]:
    """Convert Berksons to probability.

    Parameters
    ----------
    b : array_like
        Berkson values (log-odds base 2).

    Returns
    -------
    p : ndarray
        Corresponding probability values.
    d : ndarray
        Corresponding derivatives dP/dB.
    """
    b = np.asarray(b, dtype=float)
    p = 1.0 - 1.0 / (1.0 + np.power(2.0, b))
    d = np.log(2.0) * p * (1.0 - p)
    return p, d
