"""V_USASI - Generate USASI noise."""

import numpy as np
from pyvoicebox.v_randfilt import v_randfilt


def v_usasi(n, fs=8000):
    """Generate n samples of USASI noise at sample frequency fs.

    USASI noise simulates long-term average audio program material.

    Parameters
    ----------
    n : int
        Number of samples.
    fs : float, optional
        Sample frequency in Hz. Default 8000.

    Returns
    -------
    x : ndarray
        USASI noise signal.
    """
    b = np.array([1.0, 0.0, -1.0])
    a = np.real(np.poly(np.exp(-np.array([100, 320]) * 2 * np.pi / fs)))
    x, _, _, _ = v_randfilt(b, a, n)
    return x
