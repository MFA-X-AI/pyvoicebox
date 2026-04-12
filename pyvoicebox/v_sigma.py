"""V_SIGMA - Estimate glottal opening and closing instants using SIGMA algorithm.

This function requires SWT (Stationary Wavelet Transform) which is complex to implement.
A simplified stub is provided that raises NotImplementedError.
"""

from __future__ import annotations

def v_sigma(lx, fs, fmax=400) -> None:
    """Estimate glottal opening and closing instants (SIGMA algorithm).

    This function requires the Stationary Wavelet Transform (SWT) which
    is available in MATLAB's Wavelet Toolbox but not straightforwardly
    in scipy.

    Parameters
    ----------
    lx : array_like
        LX (laryngograph) signal.
    fs : float
        Sampling frequency in Hz.
    fmax : float, optional
        Maximum laryngeal frequency. Default 400 Hz.

    Raises
    ------
    NotImplementedError
        SWT-based SIGMA algorithm requires specialized wavelet toolbox support.
    """
    raise NotImplementedError(
        "v_sigma requires SWT (Stationary Wavelet Transform) which is not "
        "readily available in scipy. Consider using pywt.swt from PyWavelets."
    )
