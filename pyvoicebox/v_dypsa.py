"""V_DYPSA - Derive glottal closure instances from speech using the DYPSA algorithm."""

from __future__ import annotations
import numpy as np
from .v_lpcauto import v_lpcauto
from .v_lpcifilt import v_lpcifilt
from .v_zerocros import v_zerocros


def v_dypsa(s, fs) -> tuple[np.ndarray, np.ndarray]:
    """Derive glottal closure instances from speech.

    Parameters
    ----------
    s : array_like
        Speech signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    gci : ndarray
        Vector of glottal closure sample numbers (0-based).
    goi : ndarray
        Vector of glottal opening sample numbers (0-based).

    References
    ----------
    [1] Naylor et al., IEEE Trans Speech Audio Proc, 15:34-43, 2007.
    """
    s = np.asarray(s, dtype=float).ravel()
    ns = len(s)

    # Algorithm parameters
    cpfrac = 0.3
    fxmax = 500
    fxmin = 50
    lpcdur = 0.020
    lpcstep = 0.010
    lpcn = 2
    lpcnf = 0.001
    preemph_freq = 50
    gwlen = 0.003
    fwlen = 0.00045

    # Pre-emphasis
    alpha = np.exp(-2 * np.pi * preemph_freq / fs)
    sp = np.concatenate([[s[0]], s[1:] - alpha * s[:-1]])

    # LPC analysis parameters
    lpc_order = int(round(lpcnf * fs) + lpcn)
    lpc_frame = int(round(lpcdur * fs))
    lpc_step = int(round(lpcstep * fs))

    # Compute LPC residual
    n_frames = max(1, int(np.floor((ns - lpc_frame) / lpc_step)) + 1)
    residual = np.zeros(ns)

    for i in range(n_frames):
        start = i * lpc_step
        end = min(start + lpc_frame, ns)
        seg = sp[start:end]
        if len(seg) < lpc_order + 1:
            continue
        ar, *_ = v_lpcauto(seg, lpc_order)
        if ar.ndim > 1:
            ar = ar[0, :]
        filt_out = v_lpcifilt(ar, sp[start:min(start + lpc_frame + lpc_step, ns)])
        end_res = min(start + len(filt_out), ns)
        residual[start:end_res] = filt_out[:end_res - start]

    # Group delay computation
    gw = int(round(gwlen * fs))
    if gw < 2:
        gw = 2
    fw = int(round(fwlen * fs))
    if fw < 1:
        fw = 1

    # Compute group delay function using the residual
    nfft = int(2 ** np.ceil(np.log2(2 * gw)))
    gdwav = np.zeros(ns)
    half_gw = gw // 2

    for i in range(half_gw, ns - half_gw):
        seg = residual[i - half_gw:i + half_gw]
        if len(seg) < 2 * half_gw:
            continue
        win = np.hamming(len(seg))
        # Weighted average of derivative
        n_idx = np.arange(len(seg)) - half_gw
        gdwav[i] = np.sum(seg * win * n_idx) / (np.sum(seg * win) + 1e-20)

    # Smooth group delay
    if fw > 0:
        kernel = np.ones(fw) / fw
        gdwav = np.convolve(gdwav, kernel, mode='same')

    # Find negative-going zero crossings (candidates for GCI)
    min_period = int(round(fs / fxmax))
    max_period = int(round(fs / fxmin))

    # Find zero crossings of group delay
    zc, _ = v_zerocros(gdwav, 'n')  # negative-going
    if len(zc) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Filter candidates by minimum spacing
    gci = []
    last = -max_period
    for z in zc:
        if z - last >= min_period:
            gci.append(int(z))
            last = z

    gci = np.array(gci, dtype=int)

    # Estimate glottal openings
    goi = np.round(gci - cpfrac * np.concatenate([[max_period], np.diff(gci)])).astype(int)
    goi = np.clip(goi, 0, ns - 1)

    return gci, goi
