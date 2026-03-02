"""V_PPMVU - Calculate PPM and VU meter readings (stub).

The full implementation requires specific filter design. A simplified stub is provided.
"""

import numpy as np


def v_ppmvu(sp, fs, mode=''):
    """Calculate PPM and VU meter readings.

    Simplified implementation. The full MATLAB version implements
    detailed PPM and VU metering standards.

    Parameters
    ----------
    sp : array_like
        Input signal.
    fs : float
        Sample frequency in Hz.
    mode : str, optional
        Mode string.

    Returns
    -------
    lev : float
        Level reading.
    """
    sp = np.asarray(sp, dtype=float).ravel()
    if len(sp) == 0:
        return 0.0

    # Simple RMS level
    rms = np.sqrt(np.mean(sp ** 2))
    if rms > 0:
        return 20 * np.log10(rms)
    return -np.inf
