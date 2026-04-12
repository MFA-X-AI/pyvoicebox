"""V_FRQ2BARK - Convert Hertz to BARK frequency scale."""

from __future__ import annotations
import numpy as np


def v_frq2bark(f, m='') -> tuple[np.ndarray, np.ndarray]:
    """Convert frequencies in Hz to the BARK scale.

    Parameters
    ----------
    f : array_like
        Frequencies in Hz.
    m : str, optional
        Mode string with option characters:
        'h' - use high frequency correction from Traunmuller (1990)
        'l' - use low frequency correction from Traunmuller (1990)
        'H' - do not apply any high frequency correction
        'L' - do not apply any low frequency correction
        'z' - use the expressions from Zwicker et al. (1980)
        's' - use the expression from Schroeder et al. (1979)
        'u' - unipolar version: do not force b to be an odd function

    Returns
    -------
    b : ndarray
        Bark-scale values.
    c : ndarray
        Critical bandwidth: d(freq)/d(bark).

    References
    ----------
    [1] H. Traunmuller, "Analytical Expressions for the Tonotopic Sensory
        Scale", J. Acoust. Soc. Am. 88, 1990, pp. 97-100.
    [2] E. Zwicker, "Subdivision of the audible frequency range into critical
        bands", J Acoust Soc Am 33, 1961, p248.
    [3] M. R. Schroeder, B. S. Atal, and J. L. Hall, 1979.
    [4] E. Zwicker and E. Terhardt, 1980.
    """
    f = np.asarray(f, dtype=float)

    A = 26.81
    B = 1960.0
    C = -0.53
    D = A * B
    P = 0.53 / (3.53 ** 2)
    Q = 0.25
    R = 20.4
    xy = 2.0
    S = 0.5 * Q / xy
    T = R + 0.5 * xy
    U = T - xy

    if 'u' in m:
        g = f.copy()
    else:
        g = np.abs(f)

    if 'z' in m:
        b = 13.0 * np.arctan(0.00076 * g) + 3.5 * np.arctan((g / 7500.0) ** 2)
        c = 25.0 + 75.0 * (1.0 + 1.4e-6 * g ** 2) ** 0.69
    elif 's' in m:
        b = 7.0 * np.log(g / 650.0 + np.sqrt(1.0 + (g / 650.0) ** 2))
        c = np.cosh(b / 7.0) * 650.0 / 7.0
    else:
        b = A * g / (B + g) + C
        d = D * (B + g) ** (-2)
        if 'l' in m:
            m1 = b < 2
            d = np.where(m1, d * 0.85, d)
            b = np.where(m1, 0.3 + 0.85 * b, b)
        elif 'L' not in m:
            m1 = b < 3
            # MATLAB updates b first, then uses updated b for d
            b_new = b + P * (3 - b) ** 2
            b = np.where(m1, b_new, b)
            d = np.where(m1, d * (1 - 2 * P * (3 - b)), d)
        if 'h' in m:
            m1 = b > 20.1
            d = np.where(m1, d * 1.22, d)
            b = np.where(m1, 1.22 * b - 4.422, b)
        elif 'H' not in m:
            m2 = b > T
            m1 = (b > U) & ~m2
            # MATLAB updates b(m1) first, then b(m2), then d(m2), then d(m1)
            b = np.where(m1, b + S * (b - U) ** 2, b)
            b = np.where(m2, (1 + Q) * b - Q * R, b)
            d = np.where(m2, d * (1 + Q), d)
            d = np.where(m1, d * (1 + 2 * S * (b - U)), d)
        c = 1.0 / d

    if 'u' not in m:
        b = b * np.sign(f)

    return b, c
