"""V_ATAN2SC - sin and cosine of atan(y/x)."""

from __future__ import annotations
import numpy as np


def v_atan2sc(y, x) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sin and cosine of atan(y/x) robustly.

    Parameters
    ----------
    y, x : array_like
        Input values (same shape).

    Returns
    -------
    s : ndarray
        sin(t) where tan(t) = y/x
    c : ndarray
        cos(t) where tan(t) = y/x
    r : ndarray
        sqrt(x^2 + y^2)
    t : ndarray
        atan2(y, x)
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    s = np.zeros_like(y)
    c = np.full_like(y, np.nan)
    r = np.zeros_like(y)
    t = np.full_like(y, np.nan)

    # Handle y == 0
    m = (y == 0)
    if np.any(m):
        neg_x = (x[m] < 0).astype(float)
        t[m] = neg_x * np.pi
        c[m] = 1.0 - 2.0 * neg_x
        r[m] = np.abs(x[m])

    # Handle |y| > |x| and not yet computed
    m2 = (np.abs(y) > np.abs(x)) & np.isnan(c)
    if np.any(m2):
        q = x[m2] / y[m2]
        u = np.sqrt(1.0 + q ** 2) * np.sign(y[m2])
        s[m2] = 1.0 / u
        c[m2] = s[m2] * q
        r[m2] = y[m2] * u

    # Handle remaining (|x| >= |y|, y != 0)
    m3 = np.isnan(c)
    if np.any(m3):
        q = y[m3] / x[m3]
        u = np.sqrt(1.0 + q ** 2) * np.sign(x[m3])
        c[m3] = 1.0 / u
        s[m3] = c[m3] * q
        r[m3] = x[m3] * u

    # Compute t where still NaN
    m4 = np.isnan(t)
    if np.any(m4):
        t[m4] = np.arctan2(s[m4], c[m4])

    return s, c, r, t
