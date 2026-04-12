"""V_MIDI2FRQ - Convert musical note numbers to frequencies."""

from __future__ import annotations
import numpy as np


def v_midi2frq(n, s='e') -> np.ndarray:
    """Convert MIDI note numbers to frequencies in Hz.

    Parameters
    ----------
    n : array_like
        MIDI note numbers. Middle C is 60. Note 69 = 440 Hz.
    s : str, optional
        Scale type:
        'e' - equal tempered (default)
        'p' - Pythagorean scale
        'j' - just intonation

    Returns
    -------
    f : ndarray
        Frequencies in Hz.
    """
    n = np.asarray(n, dtype=float)

    if s and s[0] == 'p':
        r = np.array([256/243, 9/8, 32/27, 81/64, 4/3, 729/512,
                       3/2, 128/81, 27/16, 16/9, 243/128])
    elif s and s[0] == 'j':
        r = np.array([16/15, 9/8, 6/5, 5/4, 4/3, 36/25,
                       3/2, 8/5, 5/3, 9/5, 15/8])
    else:
        r = None

    if r is not None:
        # MATLAB: c=[0 0 12*log(r)/log(2)-(1:11) 0] has 14 elements (1-based indices 1..14)
        # Python: c has 14 elements (0-based indices 0..13)
        c = np.zeros(14)
        c[2:13] = 12.0 * np.log(r) / np.log(2.0) - np.arange(1, 12)
        # c[0]=0, c[1]=0, c[2..12]=corrections, c[13]=0
        nm = np.mod(n, 12.0)
        na = np.floor(nm).astype(int)
        nb = nm - na
        # MATLAB: c(na+2) and c(na+3) with 1-based indexing
        # Python: c[na+1] and c[na+2] with 0-based indexing
        f = 440.0 * np.exp((n + c[na + 1] * (1.0 - nb) + c[na + 2] * nb - 69.0) * np.log(2.0) / 12.0)
    else:
        f = 440.0 * np.exp((n - 69.0) * np.log(2.0) / 12.0)

    return f
