"""V_LPCLO2RF - Convert log area ratios to reflection coefficients."""

from __future__ import annotations
import numpy as np


def v_lpclo2rf(lo) -> np.ndarray:
    """Convert log area ratios to reflection coefficients.

    Parameters
    ----------
    lo : array_like
        Log area ratios.

    Returns
    -------
    rf : ndarray
        Reflection coefficients.
    """
    lo = np.asarray(lo, dtype=float)
    rf = -np.tanh(lo / 2)
    return rf
