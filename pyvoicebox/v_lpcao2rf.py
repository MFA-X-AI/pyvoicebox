"""V_LPCAO2RF - Convert area ratios to reflection coefficients."""

from __future__ import annotations
import numpy as np


def v_lpcao2rf(ao) -> np.ndarray:
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
