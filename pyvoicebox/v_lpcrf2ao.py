"""V_LPCRF2AO - Convert reflection coefficients to area ratios."""

import numpy as np


def v_lpcrf2ao(rf):
    """Convert reflection coefficients to area ratios.

    Parameters
    ----------
    rf : array_like
        Reflection coefficients.

    Returns
    -------
    ao : ndarray
        Area ratios.
    """
    rf = np.asarray(rf, dtype=float)
    ao = (1 - rf) / (1 + rf)
    return ao
