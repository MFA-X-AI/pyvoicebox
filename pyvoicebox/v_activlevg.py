"""V_ACTIVLEVG - Measure active speech level robustly."""

from __future__ import annotations
from pyvoicebox.v_activlev import v_activlev


def v_activlevg(sp, fs, mode='') -> tuple[np.ndarray, np.ndarray]:
    """Measure active speech level robustly.

    This is a simplified wrapper around v_activlev. The full MATLAB
    implementation uses a Gaussian mixture model approach for robust
    estimation in noisy conditions.

    Parameters
    ----------
    sp : array_like
        Speech signal.
    fs : float
        Sample frequency in Hz.
    mode : str, optional
        Mode string (same as v_activlev).

    Returns
    -------
    lev : float
        Active speech level.
    af : float
        Activity factor.
    """
    return v_activlev(sp, fs, mode)
