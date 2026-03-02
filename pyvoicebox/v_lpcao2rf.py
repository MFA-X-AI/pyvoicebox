"""V_LPCAO2RF - Convert area ratios to reflection coefficients."""

import numpy as np


def v_lpcao2rf(ao):
    """Convert area ratios to reflection coefficients.

    Parameters
    ----------
    ao : array_like
        Area ratios.

    Returns
    -------
    rf : ndarray
        Reflection coefficients.
    """
    ao = np.asarray(ao, dtype=float)
    rf = (1 - ao) / (1 + ao)
    return rf
