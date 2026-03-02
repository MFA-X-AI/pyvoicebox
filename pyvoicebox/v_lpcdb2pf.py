"""V_LPCDB2PF - Convert decibel power spectrum to power spectrum."""

import numpy as np


def v_lpcdb2pf(db):
    """Convert decibel power spectrum to power spectrum.

    Parameters
    ----------
    db : array_like
        Power spectrum in dB.

    Returns
    -------
    pf : ndarray
        Power spectrum (linear scale).
    """
    db = np.asarray(db, dtype=float)
    pf = np.exp(db * np.log(10) / 10)
    return pf
