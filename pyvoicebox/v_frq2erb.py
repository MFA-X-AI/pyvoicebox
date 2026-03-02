"""V_FRQ2ERB - Convert Hertz to ERB frequency scale."""

import numpy as np


def v_frq2erb(frq):
    """Convert frequencies in Hz to the ERB-rate scale.

    Parameters
    ----------
    frq : array_like
        Frequencies in Hz.

    Returns
    -------
    erb : ndarray
        ERB-rate scale values.
    bnd : ndarray
        ERB bandwidth in Hz.

    Notes
    -----
    The ERB scale is measured using the notched-noise method. The
    Equivalent Rectangular Bandwidth is:
        df/de = 6.23*f^2 + 93.39*f + 28.52 (f in kHz)

    References
    ----------
    [1] B.C.J. Moore & B.R. Glasberg, "Suggested formula for calculating
        auditory-filter bandwidth and excitation patterns", J. Acoust. Soc.
        Am., V74, pp 750-753, 1983.
    """
    frq = np.asarray(frq, dtype=float)

    u = np.array([6.23e-6, 93.39e-3, 28.52])
    p = np.sort(np.roots(u))  # p=[-14678.5, -311.9]
    a = 1e6 / (6.23 * (p[1] - p[0]))  # a=11.17
    c = p[0]  # c=-14678.5
    k = p[0] - p[0] ** 2 / p[1]  # k=676170.42
    h = p[0] / p[1]  # h=47.065

    g = np.abs(frq)
    erb = a * np.sign(frq) * np.log(h - k / (g - c))
    bnd = np.polyval(u, g)

    return erb, bnd
