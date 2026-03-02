"""V_VONMISESPDF - Von Mises probability distribution."""

import numpy as np
from scipy.special import i0


def v_vonmisespdf(x, m, k):
    """Von Mises probability distribution.

    Parameters
    ----------
    x : array_like
        Input values (in radians).
    m : float
        Mean angle (in radians).
    k : float
        Concentration parameter.

    Returns
    -------
    p : ndarray
        Probability density values (same shape as x).
    """
    x = np.asarray(x, dtype=float)
    p = np.exp(k * np.cos(x - m)) / (2 * np.pi * i0(k))
    return p
