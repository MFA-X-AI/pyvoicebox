"""V_ENFRAME - Split signal into (overlapping) frames: one per row."""

from __future__ import annotations
import numpy as np


def v_enframe(x, win=None, hop=None, m='', fs=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split signal up into (overlapping) frames: one per row.

    Parameters
    ----------
    x : array_like
        Input signal (1-D).
    win : int or array_like, optional
        Window or window length in samples. Default is len(x).
    hop : int or float, optional
        Frame increment in samples. If < 1, fraction of window length.
        Default is window length (non-overlapping).
    m : str, optional
        Mode string:
          'z' - zero pad final frame
          'r' - reflect last few samples for final frame
          'A' - t output as centre of mass
          'E' - t output as centre of energy
          'f' - 1-sided DFT on each frame
          'F' - 2-sided DFT on each frame
          'p' - 1-sided power spectrum
          'P' - 2-sided power spectrum
          'a' - scale window to give unity gain with overlap-add
          's' - scale so power is preserved
          'S' - scale so total energy is preserved
          'd' - make 's'/'S' give power/energy per Hz
    fs : float, optional
        Sample frequency (only needed for 'd' option). Default is 1.

    Returns
    -------
    f : ndarray
        Enframed data, one frame per row.
    t : ndarray (if requested via tuple unpacking)
        Fractional time in samples at the centre of each frame.
        First sample is index 1 (MATLAB convention).
    w : ndarray (if requested via tuple unpacking)
        Window function used, including scaling.
    """
    x = np.asarray(x, dtype=float).ravel()
    nx = len(x)

    if win is None:
        win = nx
    win = np.asarray(win, dtype=float).ravel()

    nwin = len(win)
    if nwin == 1:
        lw = int(win[0])
        w = np.ones(lw)
    else:
        lw = nwin
        w = win.copy()

    if hop is None:
        hop = lw
    elif hop < 1:
        hop = int(round(lw * hop))
    else:
        hop = int(hop)

    # Window scaling
    wsc = 1.0
    if 'a' in m:
        wsc = np.sqrt(hop / np.dot(w, w))
    elif 'd' in m:
        if 's' in m:
            wsc = np.sqrt(1.0 / (np.dot(w, w) * fs))
        elif 'S' in m:
            wsc = np.sqrt(hop / np.dot(w, w)) / fs
    else:
        if 's' in m:
            wsc = np.sqrt(1.0 / (np.dot(w, w) * lw))
        elif 'S' in m:
            wsc = np.sqrt(hop / (np.dot(w, w) * lw))

    nli = nx - lw + hop
    nf = max(int(np.fix(nli / hop)), 0)  # number of full frames
    na = nli - hop * nf + (nf == 0) * (lw - hop)  # samples left over
    fx = ('z' in m or 'r' in m) and na > 0  # need extra row

    f = np.zeros((nf + int(fx), lw))
    indf = hop * np.arange(nf)  # (nf,)
    inds = np.arange(lw)        # (lw,)

    if fx:
        # Build index matrix for full frames
        idx = indf[:, np.newaxis] + inds[np.newaxis, :]  # (nf, lw)
        if nf > 0:
            f[:nf, :] = x[idx]
        if 'r' in m:
            ix = 1 + np.mod(nf * hop + np.arange(lw), 2 * nx)
            # MATLAB: x(ix+(ix>nx).*(2*nx+1-2*ix)), 1-based
            ix0 = ix.copy()
            mask = ix0 > nx
            ix0[mask] = 2 * nx + 1 - ix0[mask]
            f[nf, :] = x[ix0.astype(int) - 1]  # convert to 0-based
        else:
            rem_samples = nx - nf * hop
            f[nf, :rem_samples] = x[nf * hop:nx]
        nf = f.shape[0]
    else:
        if nf > 0:
            idx = indf[:, np.newaxis] + inds[np.newaxis, :]
            f[:] = x[idx]

    w = wsc * w
    if nwin > 1:
        f = f * w[np.newaxis, :]
    else:
        f = wsc * f

    # Power spectrum or DFT
    ml = m.lower()
    if 'p' in ml:
        f = np.fft.fft(f, axis=1)
        f = np.real(f * np.conj(f))
        if 'p' in m:  # 1-sided
            imx = int(np.fix((lw + 1) / 2))
            f[:, 1:imx] = f[:, 1:imx] + f[:, lw - 1:lw - imx:-1]
            f = f[:, :int(np.fix(lw / 2)) + 1]
    elif 'f' in ml:
        f = np.fft.fft(f, axis=1)
        if 'f' in m:  # 1-sided
            f = f[:, :int(np.fix(lw / 2)) + 1]

    # Time output
    if 'E' in m:
        t0 = np.sum(np.arange(1, lw + 1) * w ** 2) / np.sum(w ** 2)
    elif 'A' in m:
        t0 = np.sum(np.arange(1, lw + 1) * w) / np.sum(w)
    else:
        t0 = (1 + lw) / 2.0
    t = t0 + hop * np.arange(nf)

    return f, t, w
