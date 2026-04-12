"""V_BESSELRATIOI - Inverse Bessel function ratio."""

from __future__ import annotations
import numpy as np
from .v_besselratio import v_besselratio


def v_besselratioi(r, v=0, p=5) -> np.ndarray:
    """Calculate the inverse Bessel function ratio.

    Given r = besseli(v+1,s)/besseli(v,s), find s.

    Parameters
    ----------
    r : array_like
        Value of the Bessel function ratio.
    v : int, optional
        Denominator Bessel function order (default 0, must be 0 for now).
    p : int, optional
        Digits precision <=13 (default 5).

    Returns
    -------
    s : ndarray
        Value of s such that r = besseli(v+1,s)/besseli(v,s).
    """
    p = min(max(int(np.ceil(p)), 1), 13)
    if v != 0:
        raise ValueError('v must be zero (for now)')

    r = np.asarray(r, dtype=float)
    sr = r.shape
    r = r.ravel()
    nr = r.size

    s = np.full(nr, -1.0)
    y = 2.0 / (1.0 - r)

    m1 = (r >= 0) & (r < 1)  # valid range
    mn = np.copy(m1)  # Newton iteration needed

    # Use inverse taylor series for 0 <= r <= 0.85
    m = m1 & (r <= 0.85)
    if np.any(m):
        rm = r[m]
        mn[m] = rm >= 0.642
        xm = rm ** 2
        sm = ((rm - 5.6076) * rm + 5.0797) * rm - 4.6494
        sm = sm * y[m] * xm - 1.0
        s[m] = ((((sm * xm + 15.0) * xm + 60.0) * xm / 360.0 + 1.0) * xm - 2.0) * rm / (xm - 1.0)

    # Use continued fraction for 0.85 < r < 1
    m = m1 & ~(r <= 0.85)
    if np.any(m):
        rm = r[m].copy()
        mn[m] = rm < 0.95
        ym = y[m]
        mc = rm < 0.95
        if np.any(mc):
            rm[mc] = (-2326.0 * rm[mc] + 4317.5526) * rm[mc] - 2001.035224
        if np.any(~mc):
            rm[~mc] = 32.0 / (120.0 * rm[~mc] - 131.5 + ym[~mc])
        s[m] = (ym + 1.0 + 3.0 / (ym - 5.0 - 12.0 / (ym - 10.0 - rm))) * 0.25

    # Newton iterations
    if np.any(mn):
        rmn = r[mn]
        smn = s[mn]
        ymn = y[mn]
        ymn = ((0.00048 * ymn - 0.1589) * ymn + 0.744) * ymn - 4.2932
        smn = smn + (v_besselratio(smn, 0, p + 1) - rmn) * ymn
        mr = (rmn >= 0.75) & (rmn <= 0.875)
        if np.any(mr):
            smn[mr] = smn[mr] + (v_besselratio(smn[mr], 0, p + 1) - rmn[mr]) * ymn[mr]
        s[mn] = smn

    s = s.reshape(sr)
    return s
