"""V_FRQ2CENT - Convert Hertz to Cents frequency scale."""

import numpy as np


def v_frq2cent(frq):
    """Convert frequencies in Hz to the cents scale.

    Parameters
    ----------
    frq : array_like
        Frequencies in Hz.

    Returns
    -------
    c : ndarray
        Cents scale values. 100 cents = one semitone. 440 Hz = 5700 cents.
    cr : ndarray
        Gradient in Hz/cent.

    Notes
    -----
    c = 1200 * log2(f / (440 * 2^((3/12) - 5)))

    References
    ----------
    [1] Ellis, A. "On the Musical Scales of Various Nations", Journal of
        the Society of Arts, 1885.
    """
    frq = np.asarray(frq, dtype=float)
    p = 1200.0 / np.log(2.0)
    q = 5700.0 - p * np.log(440.0)
    af = np.abs(frq)
    c = np.sign(frq) * (p * np.log(af) + q)
    cr = af / p
    return c, cr
