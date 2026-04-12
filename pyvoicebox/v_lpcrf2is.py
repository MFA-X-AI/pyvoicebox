"""V_LPCRF2IS - Convert reflection coefficients to inverse sines."""

from __future__ import annotations
import numpy as np


def v_lpcrf2is(rf) -> np.ndarray:
    """Convert reflection coefficients to inverse sines.

    Parameters
    ----------
    rf : array_like
        Reflection coefficients.

    Returns
    -------
    is_coef : ndarray
        Inverse sine coefficients.
    """
    rf = np.asarray(rf, dtype=float)
    is_coef = np.arcsin(rf) * 2 / np.pi
    return is_coef
