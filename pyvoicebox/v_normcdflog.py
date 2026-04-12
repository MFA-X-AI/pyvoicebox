"""V_NORMCDFLOG - Log of normal CDF, accurate for large negative values."""

from __future__ import annotations
import numpy as np
from scipy.special import erfc


def v_normcdflog(x, m=None, s=None) -> np.ndarray:
    """Calculate log of Normal Cumulative Distribution function.

    Parameters
    ----------
    x : array_like
        Input data.
    m : float, optional
        Mean of Normal distribution (default 0).
    s : float, optional
        Std deviation of Normal distribution (default 1).

    Returns
    -------
    p : ndarray
        log(normcdf(x)); same shape as x.
    """
    x = np.asarray(x, dtype=float)
    if s is not None:
        x = (x - m) / s
    elif m is not None:
        x = x - m

    a = 0.996
    b = -22.2491306156561  # precalculated cutoff

    t = x < b
    p = np.zeros_like(x)
    p[~t] = np.real(np.log(0.5 * erfc(-x[~t] * np.sqrt(0.5))))
    p[t] = -0.5 * (x[t] ** 2 + np.log(2 * np.pi)) - np.real(
        np.log(-x[t] - a / x[t])
    )
    return p
