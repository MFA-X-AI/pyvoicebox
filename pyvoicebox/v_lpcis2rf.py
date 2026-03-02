"""V_LPCIS2RF - Convert inverse sines to reflection coefficients."""

import numpy as np


def v_lpcis2rf(is_coef):
    """Convert inverse sines to reflection coefficients.

    Parameters
    ----------
    is_coef : array_like
        Inverse sine coefficients.

    Returns
    -------
    rf : ndarray
        Reflection coefficients.
    """
    is_coef = np.asarray(is_coef, dtype=float)
    rf = np.sin(is_coef * np.pi / 2)
    return rf
