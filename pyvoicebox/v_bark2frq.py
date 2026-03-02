"""V_BARK2FRQ - Convert the BARK frequency scale to Hertz."""

import numpy as np
from .v_frq2bark import v_frq2bark


def v_bark2frq(b, m=''):
    """Convert BARK-scale values to frequencies in Hz.

    Parameters
    ----------
    b : array_like
        Bark-scale values.
    m : str, optional
        Mode string with option characters:
        'h' - use high frequency correction from Traunmuller (1990)
        'l' - use low frequency correction from Traunmuller (1990)
        'H' - do not apply any high frequency correction
        'L' - do not apply any low frequency correction
        's' - use the expression from Schroeder et al. (1979)
        'u' - unipolar version: do not force result to be an odd function

    Returns
    -------
    f : ndarray
        Frequencies in Hz.
    c : ndarray
        Critical bandwidth: d(freq)/d(bark).

    References
    ----------
    [1] H. Traunmuller, "Analytical Expressions for the Tonotopic Sensory
        Scale", J. Acoust. Soc. Am. 88, 1990, pp. 97-100.
    [2] E. Zwicker, "Subdivision of the audible frequency range into critical
        bands", J Acoust Soc Am 33, 1961, p248.
    [3] M. R. Schroeder, B. S. Atal, and J. L. Hall, 1979.
    """
    b = np.asarray(b, dtype=float)

    A = 26.81
    B = 1960.0
    C = -0.53
    E = A + C
    D = A * B
    P = 0.53 / (3.53 ** 2)
    V = 3 - 0.5 / P
    W = V ** 2 - 9
    Q = 0.25
    R = 20.4
    xy = 2.0
    S = 0.5 * Q / xy
    T = R + 0.5 * xy
    U = T - xy
    X = T * (1 + Q) - Q * R
    Y = U - 0.5 / S
    Z = Y ** 2 - U ** 2

    if 'u' in m:
        a = b.copy()
    else:
        a = np.abs(b)

    if 's' in m:
        f = 650.0 * np.sinh(a / 7.0)
    else:
        if 'l' in m:
            m1 = a < 2
            a = np.where(m1, (a - 0.3) / 0.85, a)
        elif 'L' not in m:
            m1 = a < 3
            a = np.where(m1, V + np.sqrt(np.maximum(W + a / P, 0)), a)
        if 'h' in m:
            m1 = a > 20.1
            a = np.where(m1, (a + 4.422) / 1.22, a)
        elif 'H' not in m:
            m2 = a > X
            m1 = (a > U) & ~m2
            a = np.where(m2, (a + Q * R) / (1 + Q), a)
            a = np.where(m1, Y + np.sqrt(np.maximum(Z + a / S, 0)), a)
        f = D * (E - a) ** (-1) - B

    if 'u' not in m:
        f = f * np.sign(b)

    # Compute critical bandwidth via v_frq2bark
    _, c = v_frq2bark(f, m)

    return f, c
