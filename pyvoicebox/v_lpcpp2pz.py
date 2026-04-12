"""V_LPCPP2PZ - Convert power spectrum polynomial to power spectrum zeros."""

from __future__ import annotations
import numpy as np


def v_lpcpp2pz(pp) -> np.ndarray:
    """Convert power spectrum polynomial in cos(w) to power spectrum zeros.

    Parameters
    ----------
    pp : array_like
        Power spectrum polynomial coefficients (single polynomial).

    Returns
    -------
    pz : ndarray
        Power spectrum zeros (roots of the polynomial).
    """
    pp = np.asarray(pp, dtype=complex).ravel()
    pz = np.roots(pp)
    return pz
