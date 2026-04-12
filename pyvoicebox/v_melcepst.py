"""V_MELCEPST - Calculate the mel cepstrum of a signal."""

from __future__ import annotations
import numpy as np
from .v_enframe import v_enframe
from .v_rfft import v_rfft
from .v_rdct import v_rdct
from .v_melbankm import v_melbankm


def v_melcepst(s, fs=11025, w='M', nc=12, p=None, n=None, inc=None, fl=0, fh=0.5) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the mel cepstrum of a signal.

    Parameters
    ----------
    s : array_like
        Speech signal.
    fs : float
        Sample rate in Hz.
    w : str
        Mode string:
          'R' : rectangular window
          'N' : Hanning window
          'M' : Hamming window (default)
          'p' : filters act in power domain
          'a' : filters act in absolute magnitude domain (default)
          '0' : include 0th cepstral coefficient
          'E' : include log energy
          'd' : include delta coefficients
          'D' : include delta-delta coefficients
    nc : int
        Number of cepstral coefficients excluding 0th.
    p : int, optional
        Number of filters in filterbank. Default: floor(3*log(fs)).
    n : int, optional
        FFT length. Default: power of 2 < 0.03*fs.
    inc : int, optional
        Frame increment. Default: n//2.
    fl : float
        Low end of lowest filter as fraction of fs.
    fh : float
        High end of highest filter as fraction of fs.

    Returns
    -------
    c : ndarray
        Mel cepstrum output (one frame per row).
    tc : ndarray
        Time of each frame centre in samples.
    """
    s = np.asarray(s, dtype=float).ravel()

    if w is None or w == '':
        w = 'M'
    if p is None:
        p = int(np.floor(3 * np.log(fs)))
    if n is None:
        n = int(2 ** np.floor(np.log2(0.03 * fs)))
    if inc is None:
        inc = n // 2

    # Create window and enframe
    if 'R' in w:
        z, tc, *_ = v_enframe(s, n, inc)
    elif 'N' in w:
        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1, n + 1) / (n + 1))
        z, tc, *_ = v_enframe(s, win, inc)
    else:
        # Hamming window
        win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
        z, tc, *_ = v_enframe(s, win, inc)

    # Compute FFT
    f = v_rfft(z.T, n)  # columns are frames
    # f shape: (1+n//2, nframes)

    # Get mel filterbank
    m_bank, _, a_idx, b_idx = v_melbankm(p, n, fs, fl, fh, w)

    # Extract relevant FFT bins
    pw = f[a_idx - 1:b_idx, :] * np.conj(f[a_idx - 1:b_idx, :])  # power spectrum
    pw = np.real(pw)
    pth = np.max(pw) * 1e-20

    if 'p' in w:
        y = np.log(np.maximum(m_bank[:, a_idx - 1:b_idx].toarray() @ pw, pth))
    else:
        ath = np.sqrt(pth)
        y = np.log(np.maximum(m_bank[:, a_idx - 1:b_idx].toarray() @ np.abs(f[a_idx - 1:b_idx, :]), ath))

    # DCT
    c = v_rdct(y).T  # one frame per row
    nf = c.shape[0]
    nc_total = nc + 1

    if p > nc_total:
        c = c[:, :nc_total]
    elif p < nc_total:
        c = np.hstack([c, np.zeros((nf, nc_total - p))])

    if '0' not in w:
        c = c[:, 1:]  # remove 0th coefficient
        nc_total -= 1

    if 'E' in w:
        log_energy = np.log(np.maximum(np.sum(pw, axis=0), pth))
        c = np.column_stack([log_energy, c])
        nc_total += 1

    # Delta coefficients
    if 'D' in w:
        vf = np.array([4, 3, 2, 1, 0, -1, -2, -3, -4]) / 60.0
        af = np.array([1, 0, -1]) / 2.0
        ww = np.ones(5, dtype=int)
        cx = np.vstack([c[ww - 1, :], c, c[nf * np.ones(5, dtype=int) - 1, :]])
        vx = np.zeros((nf + 10, nc_total))
        for col in range(nc_total):
            vx[:, col] = np.convolve(cx[:, col], vf, mode='full')[:nf + 10]
        vx = vx[8:, :]  # remove initial transient

        ax = np.zeros((nf + 2, nc_total))
        for col in range(nc_total):
            ax[:, col] = np.convolve(vx[:, col], af, mode='full')[:nf + 2]
        ax = ax[2:, :]  # remove initial transient
        vx = vx[1:nf + 1, :]  # trim vx

        if 'd' in w:
            c = np.hstack([c, vx, ax])
        else:
            c = np.hstack([c, ax])
    elif 'd' in w:
        vf = np.array([4, 3, 2, 1, 0, -1, -2, -3, -4]) / 60.0
        ww = np.ones(4, dtype=int)
        cx = np.vstack([c[ww - 1, :], c, c[nf * np.ones(4, dtype=int) - 1, :]])
        vx = np.zeros((nf + 8, nc_total))
        for col in range(nc_total):
            vx[:, col] = np.convolve(cx[:, col], vf, mode='full')[:nf + 8]
        vx = vx[8:, :]
        c = np.hstack([c, vx])

    return c, tc
