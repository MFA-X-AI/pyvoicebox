"""V_PHON2SONE - Convert PHON loudness values to SONEs."""

import numpy as np


def v_phon2sone(p):
    """Convert PHON loudness values to SONEs.

    Parameters
    ----------
    p : array_like
        Matrix of phon values.

    Returns
    -------
    s : ndarray
        Matrix of sone values, same shape as p.

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
    p = np.asarray(p, dtype=float)
    b = np.log(10) * 0.1 * 0.27  # 0.27 is the exponent from [1] and [2]
    d = np.exp(b * 2.4)           # 2.4 dB is the hearing threshold from [2]
    a = 1.0 / (np.exp(b * 40) - d)  # scale factor to make p=40 give s=1
    s = a * (np.exp(b * p) - d)
    return s
