"""V_WINDINFO - Window information and figures of merit."""

from __future__ import annotations
import numpy as np


def v_windinfo(w, fs=1) -> dict:
    """Calculate window information and figures of merit.

    Parameters
    ----------
    w : array_like
        Window vector.
    fs : float, optional
        Sampling frequency. Default 1.

    Returns
    -------
    info : dict
        Dictionary with window properties:
        - nw: window length
        - ewgdelay: energy centroid delay from first sample
        - dcgain: DC gain in dB
        - enbw: equivalent noise bandwidth
        - scallop: scalloping loss in dB
    """
    w = np.asarray(w, dtype=float).ravel()
    nw = len(w)

    # DC gain
    dc = np.sum(w)
    dcgain = 20 * np.log10(abs(dc)) if dc != 0 else -np.inf

    # Energy centroid delay
    energy = np.sum(w ** 2 * np.arange(nw))
    total_energy = np.sum(w ** 2)
    ewgdelay = energy / total_energy if total_energy > 0 else 0

    # Equivalent noise bandwidth (normalized)
    enbw = nw * total_energy / (dc ** 2) if dc != 0 else np.inf

    # Scalloping loss
    nfft = max(1024, 4 * nw)
    W = np.fft.fft(w, nfft)
    W_half = np.abs(W[:nfft // 2 + 1])
    if W_half[0] > 0:
        # Response at half-bin spacing
        w_shifted = w * np.exp(1j * np.pi * np.arange(nw) / nw)
        W_shift = np.abs(np.sum(w_shifted))
        scallop = 20 * np.log10(W_shift / abs(dc)) if abs(dc) > 0 else -np.inf
    else:
        scallop = 0

    info = {
        'nw': nw,
        'ewgdelay': ewgdelay,
        'dcgain': dcgain,
        'enbw': enbw,
        'scallop': scallop,
    }
    return info
