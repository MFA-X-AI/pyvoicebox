"""V_PCMA2LIN - Convert A-law PCM to linear."""

import numpy as np


def v_pcma2lin(p, m=85, s=None):
    """Convert A-law PCM values to linear signal.

    Parameters
    ----------
    p : array_like
        A-law PCM values in the range 0 to 255.
    m : int, optional
        XOR mask applied to input values. Default is 85.
    s : float, optional
        Scale factor for output division. Default follows ITU G.711
        (equivalent to s = 2017.396342).

        Common scale factors:
            s=1       : output range +-4032
            s=2017.40 : output range +-1.998616 (default, 0 dBm0)
            s=4032    : output range +-1
            s=4096    : output range +-0.984375

    Returns
    -------
    x : ndarray
        Linear signal values.
    """
    p = np.asarray(p, dtype=float)

    if s is None:
        t = 4.95688418e-4
    else:
        t = 1.0 / s

    if m:
        q = np.bitwise_xor(p.astype(int), m).astype(float)
    else:
        q = p.copy()

    k = np.mod(q, 16)
    g = np.floor(q / 128)
    e = (q - k - 128 * g) / 16
    f = (np.abs(e - 1) - e + 1) / 2.0
    # pow2(k+16.5, e) = (k+16.5) * 2^e
    e_int = e.astype(int)
    x = (2 * g - 1) * (np.ldexp(k + 16.5, e_int) + f * (k - 15.5)) * t
    return x
