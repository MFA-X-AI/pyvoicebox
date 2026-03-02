"""V_LPCPZ2ZZ - Power spectrum roots to LPC poles."""

import numpy as np


def v_lpcpz2zz(pz):
    """Convert power spectrum roots to LPC z-plane poles.

    Parameters
    ----------
    pz : array_like
        Roots of the power spectrum polynomial pp(cos(w)).

    Returns
    -------
    zz : ndarray
        LPC z-plane poles.
    """
    pz = np.asarray(pz, dtype=complex)
    zs = np.sqrt(pz ** 2 - 1)
    zz = pz - np.sign(np.real(np.conj(pz) * zs)) * zs
    return zz
