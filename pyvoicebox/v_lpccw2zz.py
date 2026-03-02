"""V_LPCCW2ZZ - Power spectrum roots to LPC poles."""

import numpy as np


def v_lpccw2zz(cw):
    """Convert power spectrum cos(w) roots to LPC z-plane poles.

    Parameters
    ----------
    cw : array_like
        Roots of the power spectrum polynomial pp(cos(w)).

    Returns
    -------
    zz : ndarray
        LPC z-plane poles.
    """
    cw = np.asarray(cw, dtype=complex)
    zs = np.sqrt(cw ** 2 - 1)
    zz = cw - np.sign(np.real(np.conj(cw) * zs)) * zs
    return zz
