"""V_POTSBAND - Design 300-3400 Hz telephone bandwidth filter."""

import numpy as np


def v_potsband(fs):
    """Design filter for 300-3400 Hz telephone bandwidth.

    Parameters
    ----------
    fs : float
        Sample frequency in Hz.

    Returns
    -------
    b : ndarray
        Numerator coefficients.
    a : ndarray
        Denominator coefficients.
    """
    # s-plane zeros and poles of 3rd order Chebyshev type 2 highpass filter
    szp_z = np.array([0, 0.19892796195357j, -0.19892796195357j])
    szp_p = np.array([-0.97247143137874,
                      -0.48623571568937 + 0.86535995266875j,
                      -0.48623571568937 - 0.86535995266875j])

    # High pass: bilinear transform with frequency warping for 300 Hz
    tan_val = np.tan(300 * np.pi / fs)
    zl_z = 2.0 / (1 - szp_z * tan_val) - 1
    zl_p = 2.0 / (1 - szp_p * tan_val) - 1

    al = np.real(np.poly(zl_p))
    bl = np.real(np.poly(zl_z))
    # Adjust gain at Nyquist
    sw = np.array([1, -1, 1, -1])
    bl = bl * (al @ sw) / (bl @ sw)

    # Low pass: bilinear transform with frequency warping for 3400 Hz
    tan_val_h = np.tan(3400 * np.pi / fs)
    zh_z = 2.0 / (szp_z / tan_val_h - 1) + 1
    zh_p = 2.0 / (szp_p / tan_val_h - 1) + 1

    ah = np.real(np.poly(zh_p))
    bh = np.real(np.poly(zh_z))
    bh = bh * np.sum(ah) / np.sum(bh)

    b = np.convolve(bh, bl)
    a = np.convolve(ah, al)
    return b, a
