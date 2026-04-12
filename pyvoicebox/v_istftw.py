"""V_ISTFTW - Inverse Short-time Fourier Transform."""

from __future__ import annotations
import numpy as np


def v_istftw(y, so, io=None) -> np.ndarray:
    """Convert time-frequency domain back to time domain using inverse STFT.

    Parameters
    ----------
    y : array_like
        STFT data (frames x frequencies).
    so : dict
        Structure from v_stftw containing window and transform parameters.
    io : dict, optional
        State from previous call for chunked processing.

    Returns
    -------
    z : ndarray
        Reconstructed time-domain signal.
    io : dict
        State for subsequent calls.
    """
    y = np.asarray(y)
    nframes = y.shape[0]
    nt = so['nt']
    nh = so['nh']
    nw = so['nw']
    wa = so['wa']

    # Synthesis window (same as analysis for now)
    ws = wa.copy()

    # Compute normalization
    norm = np.zeros(nw)
    ov = so.get('ov', nw // nh)
    for k in range(ov):
        shifted = np.roll(wa * ws, k * nh)
        norm += shifted
    # Avoid division by zero
    norm[norm == 0] = 1.0

    # Output length
    out_len = (nframes - 1) * nh + nw
    z = np.zeros(out_len)
    w_sum = np.zeros(out_len)

    for i in range(nframes):
        # Inverse FFT
        frame = np.fft.irfft(y[i, :], nt)[:nw]
        frame *= ws

        start = i * nh
        z[start:start + nw] += frame
        w_sum[start:start + nw] += wa * ws

    # Normalize by window sum
    nz = w_sum > 1e-10
    z[nz] /= w_sum[nz]

    return z
