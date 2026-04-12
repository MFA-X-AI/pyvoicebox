"""V_CHIMV - Approximate mean and variance of non-central chi distribution."""

from __future__ import annotations
import numpy as np
from scipy.stats import norm


def v_chimv(n, l=0, s=1) -> tuple[np.ndarray, np.ndarray]:
    """Approximate mean and variance of non-central chi distribution.

    Parameters
    ----------
    n : int
        Degrees of freedom.
    l : float or array_like, optional
        Non-centrality parameter (default 0).
    s : float, optional
        Standard deviation of Gaussian (default 1).

    Returns
    -------
    m : ndarray
        Mean of chi distribution.
    v : ndarray
        Variance of chi distribution.
    """
    pp = np.array([0.595336298258636, -1.213013700592756, -0.018016200037799, 1.999986150447582, 0.0])
    qq = np.array([-0.161514114798972, 0.368983655790737, -0.136992134476950, -0.499681107630725, 2.0])

    l = np.asarray(l, dtype=float)
    ls = l / s
    l2 = ls ** 2
    s2 = s ** 2

    if n == 1:
        m = l * (1.0 - 2.0 * norm.cdf(-ls)) + 2.0 * s * norm.pdf(-ls)
    else:
        nab = 200
        ni = 1.0 / n
        ab_a = np.polyval(qq, ni)
        ab_b = np.polyval(pp, ni)
        m = np.sqrt(l2 + n - 1 + (ab_a + ab_b * l2) ** (-1)) * s

    v = (n + l2) * s2 - m ** 2
    return m, v
