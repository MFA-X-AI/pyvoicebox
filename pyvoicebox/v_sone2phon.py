"""V_SONE2PHON - Convert SONE loudness values to PHONs."""

import numpy as np


def v_sone2phon(s):
    """Convert SONE loudness values to PHONs.

    Parameters
    ----------
    s : array_like
        Matrix of sone values.

    Returns
    -------
    p : ndarray
        Matrix of phon values, same shape as s.

    Notes
    -----
    The phon scale measures perceived loudness in dB; at 1 kHz it is identical
    to dB SPL relative to 20e-6 Pa sound pressure. The sone scale is proportional
    to apparent loudness and, by definition, equals 1 at 40 phon.

    References
    ----------
    [1] J. Lochner and J. Burger. Form of the loudness function in the presence
        of masking noise. JASA, 33: 1705, 1961.
    [2] ISO/TC43. Acoustics Normal equal-loudness-level contours.
        Standard ISO 226:2003, Aug. 2003.
    """
    s = np.asarray(s, dtype=float)
    b = 1.0 / (np.log(10) * 0.1 * 0.27)  # 0.27 is the exponent from [1] and [2]
    d = np.exp(2.4 / b)                    # 2.4 dB is the hearing threshold from [2]
    a = np.exp(40.0 / b) - d               # scale factor to make p=40 give s=1
    p = b * np.log(a * s + d)
    return p
