"""V_OVERLAPADD - Join overlapping frames together."""

from __future__ import annotations
import numpy as np


def v_overlapadd(f, win=None, inc=None) -> np.ndarray:
    """Join overlapping frames together.

    Parameters
    ----------
    f : ndarray, shape (nr, nw)
        Frames to be added together, one frame per row.
    win : array_like or dict, optional
        Window function to multiply each frame, or saved state dict.
        If omitted, a rectangular window is used.
    inc : int, optional
        Time increment (in samples) between successive frames.
        Default is nw.

    Returns
    -------
    x : ndarray
        Output signal of length nw + (nr-1)*inc.
    zo : dict (only if explicitly requested)
        Saved state for chunk processing.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim == 1:
        f = f.reshape(1, -1)
    nr, nf = f.shape

    if win is None:
        if inc is None:
            inc = nf
        w = None
    elif isinstance(win, dict):
        w = win['w']
        inc = win['inc']
        xx = win['xx']
    else:
        win = np.asarray(win, dtype=float).ravel()
        if inc is None:
            if len(win) == 1 and win[0] == int(win[0]):
                inc = int(win[0])
                w = None
            else:
                inc = nf
                w = win
                if len(w) != nf:
                    raise ValueError('window length does not match frame size')
                if np.all(w == 1):
                    w = None
        else:
            w = win
            if len(w) != nf:
                raise ValueError('window length does not match frame size')
            if np.all(w == 1):
                w = None

    if not isinstance(win, dict):
        xx = None

    nb = int(np.ceil(nf / inc))  # number of overlap buffers
    no = nf + (nr - 1) * inc     # total output length

    z = np.zeros((no, nb))

    for i in range(nr):
        buf_idx = i % nb
        start = i * inc
        frame = f[i, :] * w if w is not None else f[i, :]
        z[start:start + nf, buf_idx] += frame

    x = np.sum(z, axis=1)

    if xx is not None and len(xx) > 0:
        x[:len(xx)] += xx

    return x
