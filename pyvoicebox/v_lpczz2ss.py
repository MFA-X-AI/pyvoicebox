"""V_LPCZZ2SS - Convert z-plane poles to s-plane poles."""

import numpy as np


def v_lpczz2ss(zz):
    """Convert z-plane poles to s-plane poles.

    Parameters
    ----------
    zz : array_like
        Z-plane poles.

    Returns
    -------
    ss : ndarray
        S-plane poles in normalized Hz units.
    """
    zz = np.asarray(zz, dtype=complex)
    ss = np.log(np.maximum(np.abs(zz), 1e-8) * np.exp(1j * np.angle(zz))) * 0.5 / np.pi
    return ss
