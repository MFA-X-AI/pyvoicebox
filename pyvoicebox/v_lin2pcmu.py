"""V_LIN2PCMU - Convert linear to Mu-law PCM.

Attempt at a faithful port of voicebox/v_lin2pcmu.m
"""

from __future__ import annotations
import numpy as np


def v_lin2pcmu(x, s=4004.189931) -> np.ndarray:
    """Convert linear signal to Mu-law PCM values.

    Parameters
    ----------
    x : array_like
        Input signal values.
    s : float, optional
        Scale factor applied to input values. Default is 4004.189931
        (ITU G.711 standard: sqrt((2207^2 + 5215^2)/2)).

        Common scale factors:
            s=1       : input range +-8159
            s=4004.19 : input range +-2.03761563 (default, 0 dBm0)
            s=8159    : input range +-1

    Returns
    -------
    p : ndarray
        Mu-law PCM values in the range 0 to 255.
    """
    x = np.asarray(x, dtype=float)
    y = x * s
    # Clip to +-8031
    y = (np.abs(y + 8031) - np.abs(y - 8031)) / 2.0
    # Sign: q=1 for positive, q=0 for negative
    q = np.floor((y + 8032) / 8032)
    # Decompose |y|+33 into mantissa and exponent (base-2 log)
    m, e = np.frexp(np.abs(y) + 33)
    # m is in [0.5, 1), e is integer such that |y|+33 = m * 2^e
    # MATLAB log2 returns same (m,e) as Python frexp
    p = 175 + 128 * q - 8 * (e + np.abs(e - 6)) - np.floor(32 * m - 16)
    return p.astype(np.float64)
