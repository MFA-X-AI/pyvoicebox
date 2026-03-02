"""V_LIN2PCMA - Convert linear PCM to A-law."""

import numpy as np


def v_lin2pcma(x, m=85, s=2017.396342):
    """Convert linear signal to A-law PCM values.

    Parameters
    ----------
    x : array_like
        Input signal values.
    m : int, optional
        XOR mask value applied to output. Default is 85.
    s : float, optional
        Scale factor applied to input values. Default is 2017.396342
        (ITU G.711 standard: sqrt((1120^2 + 2624^2)/2)).

        Common scale factors:
            s=1       : input range +-4096
            s=2017.40 : input range +-2.03033976 (default, 0 dBm0)
            s=4096    : input range +-1

    Returns
    -------
    p : ndarray
        A-law PCM values in the range 0 to 255.
    """
    x = np.asarray(x, dtype=float)
    # pow2(s, -6) = s * 2^(-6) = s / 64
    y = x * (s / 64.0)
    # Clip to +-63
    y = (np.abs(y + 63) - np.abs(y - 63)) / 2.0
    # Sign: q=1 for positive, q=0 for negative
    q = np.floor((y + 64) / 64)
    # Decompose |y| into mantissa and exponent
    a, e = np.frexp(np.abs(y))
    # a is in [0.5, 1), e is integer such that |y| = a * 2^e
    # d = max(e, 0) -- only keep non-negative exponents
    d = ((e + np.abs(e)) / 2).astype(int)
    # pow2(a, e - d + 5) = a * 2^(e - d + 5)
    p = 128 * q + 16 * d + np.floor(np.ldexp(a, e - d + 5))
    p = p.astype(int)
    if m:
        p = np.bitwise_xor(p, m)
    return p.astype(np.float64)
