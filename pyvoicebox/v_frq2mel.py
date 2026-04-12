"""V_FRQ2MEL - Convert Hertz to Mel frequency scale."""

from __future__ import annotations
import numpy as np


def v_frq2mel(frq) -> tuple[np.ndarray, np.ndarray]:
    """Convert frequencies in Hz to the Mel scale.

    Parameters
    ----------
    frq : array_like
        Frequencies in Hz.

    Returns
    -------
    mel : ndarray
        Mel-scale values. mel(1000 Hz) = 1000.
    mr : ndarray
        Gradient in Hz/mel.

    Notes
    -----
    The relationship is: m = ln(1 + f/700) * 1000 / ln(1 + 1000/700)
    This means that m(1000) = 1000.

    References
    ----------
    [1] J. Makhoul and L. Cosell. "Lpcw: An lpc vocoder with linear
        predictive spectral warping", Proc IEEE ICASSP, 1976.
    """
    frq = np.asarray(frq, dtype=float)
    k = 1000.0 / np.log(1.0 + 1000.0 / 700.0)  # 1127.01048
    af = np.abs(frq)
    mel = np.sign(frq) * np.log(1.0 + af / 700.0) * k
    mr = (700.0 + af) / k
    return mel, mr
