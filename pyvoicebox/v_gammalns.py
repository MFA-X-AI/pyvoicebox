"""V_GAMMALNS - Log of Gamma(x) for positive or negative real x."""

from __future__ import annotations
import numpy as np
from scipy.special import gammaln


def v_gammalns(x, return_sign=False) -> np.ndarray:
    """Compute log(|Gamma(x)|) and optionally sign(Gamma(x)).

    Parameters
    ----------
    x : array_like
        Real input values.
    return_sign : bool, optional
        If True, return (y, s) where s = sign(Gamma(x)).
        If False (default), return y which may be complex for negative Gamma.

    Returns
    -------
    y : ndarray
        log(|Gamma(x)|) if return_sign=True, else log(Gamma(x)) (complex where needed).
    s : ndarray (only if return_sign=True)
        sign(Gamma(x))
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    s = np.ones_like(x)

    m = x <= 0
    # Non-negative x values
    if not np.all(m):
        y[~m] = gammaln(x[~m])

    # Non-positive integers: Gamma is infinite
    f = m & (x == np.fix(x))
    if np.any(f):
        y[f] = np.inf
        m = m & ~f

    # Negative non-integer x values
    if np.any(m):
        t = np.sin(np.pi * x[m])
        if return_sign:
            p = t < 0
            s[m] = 1 - 2 * p
            y[m] = np.log(np.pi) - gammaln(1 - x[m]) - np.log(np.abs(t))
        else:
            y[m] = np.log(np.pi) - gammaln(1 - x[m]) - np.log(t + 0j)
            # Make result complex where needed
            y = y.astype(complex)

    if return_sign:
        return y, s
    return y
