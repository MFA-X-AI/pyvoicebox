"""V_MEL2FRQ - Convert Mel frequency scale to Hertz."""

from __future__ import annotations
import numpy as np


def v_mel2frq(mel) -> tuple[np.ndarray, np.ndarray]:
    """Convert Mel-scale values to frequencies in Hz.

    Parameters
    ----------
    mel : array_like
        Mel-scale values.

    Returns
    -------
    frq : ndarray
        Frequencies in Hz.
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
    mel = np.asarray(mel, dtype=float)
    k = 1000.0 / np.log(1.0 + 1000.0 / 700.0)  # 1127.01048
    frq = 700.0 * np.sign(mel) * (np.exp(np.abs(mel) / k) - 1.0)
    mr = (700.0 + np.abs(frq)) / k
    return frq, mr
