"""V_BESRATINV0 - Inverse of Modified Bessel Ratio I1(k)/I0(k)."""

from __future__ import annotations
import numpy as np
from .v_besselratio import v_besselratio


def v_besratinv0(r) -> np.ndarray:
    """Inverse function of the Modified Bessel Ratio I1(k)/I0(k).

    Parameters
    ----------
    r : array_like
        Input argument in range [0,1].

    Returns
    -------
    k : ndarray
        Output satisfying r = I1(k)/I0(k).
    """
    r = np.asarray(r, dtype=float)
    k = np.zeros_like(r)

    m1 = r <= 0.85
    a = r[m1]
    if a.size > 0:
        y = 2.0 / (1.0 - a)
        x = a * a
        s = ((a - 5.6076) * a + 5.0797) * a - 4.6494
        s = s * y * x - 1.0
        s = ((((s * x + 15.0) * x + 60.0) * x / 360.0 + 1.0) * x - 2.0) * a / (x - 1.0)

        m2 = a >= 0.642
        b = a[m2]
        if b.size > 0:
            z = y[m2]
            yy = ((0.00048 * z - 0.1589) * z + 0.744) * z - 4.2932
            t = s[m2]
            t = (v_besselratio(t, 0, 9) - b) * yy + t
            m3 = b >= 0.75
            c = b[m3]
            if c.size > 0:
                u = t[m3]
                t[m3] = (v_besselratio(u, 0, 9) - c) * yy[m3] + u
            s[m2] = t
        k[m1] = s

    a = r[~m1]
    if a.size > 0:
        y = 2.0 / (1.0 - a)
        x = np.zeros_like(a)
        m2 = a > 0.95
        x[m2] = 32.0 / (120.0 * a[m2] - 131.5 + y[m2])
        x[~m2] = (-2326.0 * a[~m2] + 4317.5526) * a[~m2] - 2001.035224
        s = (y + 1.0 + 3.0 / (y - 5.0 - 12.0 / (y - 10.0 - x))) * 0.25

        m2 = a < 0.95
        b = a[m2]
        if b.size > 0:
            z = y[m2]
            yy = ((0.00048 * z - 0.1589) * z + 0.744) * z - 4.2932
            t = s[m2]
            t = (v_besselratio(t, 0, 9) - b) * yy + t
            m3 = b <= 0.875
            c = b[m3]
            if c.size > 0:
                u = t[m3]
                t[m3] = (v_besselratio(u, 0, 9) - c) * yy[m3] + u
            s[m2] = t
        k[~m1] = s

    return k
