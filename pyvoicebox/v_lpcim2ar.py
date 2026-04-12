"""V_LPCIM2AR - Convert impulse response to AR coefficients."""

from __future__ import annotations
import numpy as np
from scipy.linalg import toeplitz, solve


def v_lpcim2ar(im) -> np.ndarray:
    """Convert impulse response to AR coefficients.

    Parameters
    ----------
    im : array_like, shape (nf, p+1)
        Impulse response.

    Returns
    -------
    ar : ndarray, shape (nf, p+1)
        Autoregressive coefficients.
    """
    im = np.atleast_2d(np.asarray(im, dtype=float))
    nf, p1 = im.shape
    ar = np.zeros((nf, p1))
    wz = np.zeros(p1)
    wz[0] = 1.0
    for k in range(nf):
        T = toeplitz(wz, im[k, :] / im[k, 0])
        ar[k, :] = solve(T.T, wz)
    return ar
