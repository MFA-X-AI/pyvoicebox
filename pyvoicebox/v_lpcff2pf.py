"""V_LPCFF2PF - Convert complex spectrum to power spectrum."""

from __future__ import annotations
import numpy as np


def v_lpcff2pf(ff) -> np.ndarray:
    """Convert complex spectrum to power spectrum.

    Parameters
    ----------
    ff : array_like
        Complex spectrum.

    Returns
    -------
    pf : ndarray
        Power spectrum.
    """
    ff = np.asarray(ff, dtype=complex)
    pf = np.abs(ff) ** 2
    return pf
