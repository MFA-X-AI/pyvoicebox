"""V_PCMU2LIN - Convert Mu-law PCM to linear."""

from __future__ import annotations
import numpy as np


def v_pcmu2lin(p, s=None) -> np.ndarray:
    """Convert Mu-law PCM values to linear signal.

    Parameters
    ----------
    p : array_like
        Mu-law PCM values in the range 0 to 255.
    s : float, optional
        Scale factor for output division. Default follows ITU G.711
        (equivalent to s = 4004.189931).

        Common scale factors:
            s=1       : output range +-8031
            s=4004.19 : output range +-2.005649 (default, 0 dBm0)
            s=8031    : output range +-1
            s=8159    : output range +-0.9843118

    Returns
    -------
    x : ndarray
        Linear signal values.
    """
    p = np.asarray(p, dtype=float)

    if s is None:
        t = 9.98953613e-4
    else:
        t = 4.0 / s

    m = 15 - np.mod(p, 16)         # MATLAB rem for non-negative is same as mod
    q = np.floor(p / 128)
    e = (127 - p - m + 128 * q) / 16
    # pow2(f, e) in MATLAB = f * 2^e = np.ldexp(f, e)
    x = (q - 0.5) * (np.ldexp(m + 16.5, e.astype(int)) - 16.5) * t
    return x
