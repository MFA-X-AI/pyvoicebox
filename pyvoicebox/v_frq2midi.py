"""V_FRQ2MIDI - Convert frequencies to musical note numbers."""

from __future__ import annotations
import numpy as np


def v_frq2midi(f) -> tuple[np.ndarray, list]:
    """Convert frequencies in Hz to MIDI note numbers.

    Parameters
    ----------
    f : array_like
        Frequencies in Hz. Negative frequencies produce flats instead of
        sharps in the text representation.

    Returns
    -------
    n : ndarray
        MIDI note numbers. Middle C is 60. Note 69 (A above middle C) = 440 Hz.
        Note numbers are not necessarily integers.
    t : list of str
        Text representation of the nearest note. E.g. 'C4 ' for middle C,
        'C4#' for C sharp. For negative frequencies, flats are used: 'D4-'.
    """
    f = np.asarray(f, dtype=float)
    n = 69.0 + 12.0 * np.log(np.abs(f) / 440.0) / np.log(2.0)

    # Text representation
    m = np.round(n).astype(int).ravel()
    f_flat = f.ravel()
    o = m // 12 - 1
    # MATLAB 1-based index; subtract 1 for Python 0-based
    m_idx = m - 12 * o + 6 * np.sign(f_flat).astype(int) - 5 - 1

    # First 12: flat names, next 12: sharp names (matching MATLAB layout)
    a_str = 'CDDEEFGGAABBCCDDEFFGGAAB'  # 24 chars
    b_str = ' - -  - - -  # #  # # # '  # 24 chars

    t = []
    for i in range(len(m)):
        idx = int(m_idx[i])
        # Clamp index to valid range (0 to 23)
        idx = max(0, min(23, idx))
        note = a_str[idx]
        octave = str(int(o[i]) % 10)
        accidental = b_str[idx]
        t.append(note + octave + accidental)

    return n, t
