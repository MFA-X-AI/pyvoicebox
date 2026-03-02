"""V_LPCRF2LO - Convert reflection coefficients to log area ratios."""

import numpy as np


def v_lpcrf2lo(rf):
    """Convert reflection coefficients to log area ratios.

    Parameters
    ----------
    rf : array_like
        Reflection coefficients.

    Returns
    -------
    lo : ndarray
        Log area ratios (limited to about +-14.5).
    """
    rf = np.asarray(rf, dtype=float)
    r = np.clip(rf, -1 + 1e-6, 1 - 1e-6)
    lo = np.log((1 - r) / (1 + r))
    return lo
