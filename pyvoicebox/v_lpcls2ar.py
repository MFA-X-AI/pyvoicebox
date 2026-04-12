"""V_LPCLS2AR - Convert line spectrum pair frequencies to AR polynomial."""

from __future__ import annotations
import numpy as np


def v_lpcls2ar(ls) -> np.ndarray:
    """Convert line spectrum pair frequencies to AR polynomial.

    Parameters
    ----------
    ls : array_like, shape (nf, p)
        Line spectrum pair frequencies (0 to 0.5 normalized Hz).

    Returns
    -------
    ar : ndarray, shape (nf, p+1)
        Autoregressive coefficients.
    """
    ls = np.atleast_2d(np.asarray(ls, dtype=float))
    nf, p = ls.shape
    p1 = p + 1
    p2 = p1 * 2
    ar = np.zeros((nf, p1))

    for k in range(nf):
        le = np.exp(ls[k, :] * np.pi * 2j)
        lf = np.concatenate([[1], le, [-1], np.conj(le[::-1])])
        y = np.real(np.poly(lf[0:p2:2]))
        x = np.real(np.poly(lf[1:p2:2]))
        ar[k, :] = (x[:p1] + y[:p1]) / 2.0

    return ar
