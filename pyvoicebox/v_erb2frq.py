"""V_ERB2FRQ - Convert ERB frequency scale to Hertz."""

import numpy as np


def v_erb2frq(erb):
    """Convert ERB-rate scale values to frequencies in Hz.

    Parameters
    ----------
    erb : array_like
        ERB-rate scale values.

    Returns
    -------
    frq : ndarray
        Frequencies in Hz.
    bnd : ndarray
        ERB bandwidth in Hz.

    Notes
    -----
    ERB values are clipped to 43.032 which corresponds to infinite frequency.

    References
    ----------
    [1] B.C.J. Moore & B.R. Glasberg, "Suggested formula for calculating
        auditory-filter bandwidth and excitation patterns", J. Acoust. Soc.
        Am., V74, pp 750-753, 1983.
    """
    erb = np.asarray(erb, dtype=float)

    u = np.array([6.23e-6, 93.39e-3, 28.52])
    p = np.sort(np.roots(u))  # p=[-14678.5, -311.9]
    d = 1e-6 * (6.23 * (p[1] - p[0]))  # d=0.0895
    c = p[0]  # c=-14678.5
    k = p[0] - p[0] ** 2 / p[1]  # k=676170.4
    h = p[0] / p[1]  # h=47.06538

    frq = np.sign(erb) * (k / np.maximum(h - np.exp(d * np.abs(erb)), 0) + c)
    bnd = np.polyval(u, np.abs(frq))

    return frq, bnd
