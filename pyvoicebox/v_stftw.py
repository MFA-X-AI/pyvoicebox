"""V_STFTW - Short-time Fourier Transform."""

from __future__ import annotations
import numpy as np
from pyvoicebox.v_windows import v_windows


def v_stftw(x, nw, m='', ov=2, nt=None) -> tuple[np.ndarray, dict]:
    """Convert time-domain signal to time-frequency domain using STFT.

    Parameters
    ----------
    x : array_like
        Input signal.
    nw : int
        Window length (rounded up to multiple of ov).
    m : str, optional
        Mode string including window code.
    ov : int, optional
        Overlap factor. Default 2.
    nt : int, optional
        DFT length. Default nw.

    Returns
    -------
    y : ndarray
        STFT output (frames x frequencies).
    so : dict
        Structure for inverse transformation.
    """
    x = np.asarray(x, dtype=float).ravel()
    nx = len(x)

    # Round nw up to multiple of ov
    nw = int(np.ceil(nw / ov) * ov)
    nh = nw // ov  # hop size

    if nt is None:
        nt = nw

    # Choose window
    if 'n' in m:
        wa = np.hanning(nw + 1)[:nw]  # Hann
    elif 'c' in m:
        wa = np.cos(np.pi * (np.arange(nw) + 0.5) / nw)
    elif 'R' in m:
        wa = np.ones(nw)
    elif 'M' in m or (ov == 2 and 'm' not in m):
        # sqrt Hamming
        wa = np.sqrt(np.hamming(nw))
    else:
        wa = np.hamming(nw)

    # Normalize for COLA
    wa_sum = np.zeros(nw)
    for k in range(ov):
        wa_sum += np.roll(wa ** 2, k * nh)
    # Don't normalize if wa_sum is already constant

    # Pad signal
    if 'e' in m:
        # Pad beginning and end
        pad_start = nw - nh
        pad_end = nw - nh
        x = np.concatenate([np.zeros(pad_start), x, np.zeros(pad_end)])
        nx = len(x)

    # Pad final frame
    if 'z' in m:
        pad = nh - (nx % nh) if nx % nh != 0 else 0
        x = np.concatenate([x, np.zeros(pad)])
    elif 'r' in m or True:
        # Pad with reflected data
        pad = nh - (nx % nh) if nx % nh != 0 else 0
        if pad > 0:
            x = np.concatenate([x, x[-1:-(pad + 1):-1]])

    nx = len(x)
    nframes = max(0, (nx - nw) // nh + 1)
    nf = nt // 2 + 1

    y = np.zeros((nframes, nf), dtype=complex)
    for i in range(nframes):
        start = i * nh
        frame = x[start:start + nw] * wa
        if nt > nw:
            frame = np.concatenate([frame, np.zeros(nt - nw)])
        Y = np.fft.rfft(frame, nt)
        y[i, :] = Y

    so = {
        'nt': nt,
        'nh': nh,
        'nw': nw,
        'wa': wa,
        'ov': ov,
    }

    return y, so
