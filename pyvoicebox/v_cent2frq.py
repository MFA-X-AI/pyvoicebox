"""V_CENT2FRQ - Convert Cents frequency scale to Hertz."""

from __future__ import annotations
import numpy as np


def v_cent2frq(c) -> tuple[np.ndarray, np.ndarray]:
    """Convert cents scale values to frequencies in Hz.

    Parameters
    ----------
    c : array_like
        Cents scale values. 100 cents = one semitone. 440 Hz = 5700 cents.

    Returns
    -------
    frq : ndarray
        Frequencies in Hz.
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
    c = np.asarray(c, dtype=float)
    p = 1200.0 / np.log(2.0)
    q = 5700.0 - p * np.log(440.0)
    af = np.exp((np.abs(c) - q) / p)
    frq = np.sign(c) * af
    cr = af / p
    return frq, cr
