"""V_CORRELOGRAM - Calculate correlogram."""

import numpy as np
from .v_windows import v_windows


def v_correlogram(x, inc=128, nw=None, nlag=None, m='h', fs=1):
    """Calculate correlogram.

    Parameters
    ----------
    x : ndarray
        Input signal (samples, channels) from a filterbank.
    inc : int
        Frame increment in samples.
    nw : int or array_like, optional
        Window length in samples or window function. Default: inc.
    nlag : int, optional
        Number of lags to calculate. Default: nw.
    m : str
        Mode: 'h' for Hamming window.
    fs : float
        Sample frequency.

    Returns
    -------
    y : ndarray
        Correlogram (nlag, channels, frames).
    ty : ndarray
        Time of window centre for each frame.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    nx, nc = x.shape

    if nw is None:
        nw = inc

    nw_arr = np.atleast_1d(np.asarray(nw, dtype=float))
    if len(nw_arr) > 1:
        win = nw_arr.ravel()
        nwin = len(win)
    else:
        nwin = int(nw_arr[0])
        if 'h' in m:
            win = v_windows(3, nwin).ravel()
        else:
            win = np.ones(nwin)

    if nlag is None:
        nlag = nwin

    nwl = nwin + nlag - 1
    nt = int(2 ** np.ceil(np.log2(nwl)))
    nf_frames = int(np.floor((nx - nwl + inc) / inc))

    if nf_frames <= 0:
        return np.zeros((nlag, nc, 0)), np.array([])

    wincg = np.dot(np.arange(1, nwin + 1), win ** 2) / np.dot(win, win)
    fwin = np.conj(np.fft.fft(win, nt))

    y = np.zeros((nlag, nc, nf_frames))

    for iframe in range(nf_frames):
        for ic in range(nc):
            start = iframe * inc
            x_seg = x[start:start + nwl, ic]
            if len(x_seg) < nwl:
                x_seg = np.concatenate([x_seg, np.zeros(nwl - len(x_seg))])

            x_win = x_seg[:nwin] * win
            X_win = np.fft.fft(x_win, nt)
            X_full = np.fft.fft(x_seg, nt)
            v = np.fft.ifft(np.conj(X_win) * X_full)

            x_sq = x_seg ** 2
            X_sq = np.fft.fft(x_sq, nt)
            w_energy = np.real(np.fft.ifft(fwin * X_sq))
            w_energy = np.maximum(w_energy[:nlag], 0)
            w0 = w_energy[0]
            norm = np.sqrt(w_energy * w0)
            norm[norm == 0] = 1.0

            if np.isreal(x).all():
                y[:, ic, iframe] = np.real(v[:nlag]) / norm
            else:
                y[:, ic, iframe] = np.real(np.conj(v[:nlag])) / norm

    ty = np.arange(nf_frames) * inc + wincg
    return y, ty
