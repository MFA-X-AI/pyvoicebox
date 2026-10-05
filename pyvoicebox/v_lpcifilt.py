"""V_LPCIFILT - Apply inverse filter to speech signal."""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter


def v_lpcifilt(s, ar, t=None, dc=None, fade=None) -> np.ndarray:
    """Apply inverse filter to speech signal.

    Parameters
    ----------
    s : array_like, shape (ns,)
        Speech signal.
    ar : array_like, shape (nf, p+1)
        AR coefficients, one row per frame.
    t : array_like, shape (nf,), optional
        Index of first sample in each frame (1-based as in MATLAB).
    dc : array_like, shape (nf,) or scalar, optional
        DC component to subtract from signal.
    fade : float, optional
        Number of samples for linear interpolation at frame boundaries.

    Returns
    -------
    u : ndarray, shape (ns,)
        Inverse filtered signal.
    """
    s = np.asarray(s, dtype=float).ravel()
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    ns = len(s)

    if dc is None:
        dc = np.zeros(nf)
    else:
        dc = np.asarray(dc, dtype=float).ravel()
        if len(dc) == 1:
            dc = np.full(nf, dc[0])

    if nf == 1:
        return lfilter(ar[0, :], [1.0], s - dc[0])

    p = p1 - 1

    if fade is None:
        fade = 0
    if t is None:
        t = p1 + np.arange(nf) * (ns - p) / nf

    t = np.asarray(t, dtype=float).ravel()

    u = np.zeros(ns)
    if fade < 1:
        # Last frame
        x0_start = max(0, int(np.ceil(t[nf - 1] - 1) - p))
        x0 = np.arange(x0_start, ns)
        if len(x0) > 0:
            u[x0] = lfilter(ar[nf - 1, :], [1.0], s[x0] - dc[nf - 1])

        # Middle frames
        for i in range(nf - 2, 0, -1):
            x0_start = max(0, int(np.ceil(t[i] - 1) - p))
            x0_end = int(np.ceil(t[i + 1] - 1))
            x0 = np.arange(x0_start, x0_end)
            if len(x0) > 0:
                u[x0] = lfilter(ar[i, :], [1.0], s[x0] - dc[i])

        # First frame
        x0 = np.arange(0, int(np.ceil(t[1] - 1)))
        if len(x0) > 0:
            u[x0] = lfilter(ar[0, :], [1.0], s[x0] - dc[0])
    else:
        # Fade intervals are clipped so adjacent transitions cannot overlap.
        tb = min(t[-1] + fade, (t[-1] + ns) / 2)
        ta = max(t[-1] - fade, (t[-2] + t[-1]) / 2)
        start = max(0, int(np.ceil(ta)) - p - 1)
        x0 = np.arange(start, ns)
        xb = x0 + 1 - tb
        weight = np.maximum(1 + (xb - np.abs(xb)) / (2 * (tb - ta)), 0)
        u[x0] = lfilter(ar[-1, :], [1.0], s[x0] - dc[-1]) * weight

        for i in range(nf - 2, 0, -1):
            td, tc = tb, ta
            tb = min(t[i] + fade, ta)
            ta = max(t[i] - fade, (t[i - 1] + t[i]) / 2)
            start = max(0, int(np.ceil(ta)) - p - 1)
            x0 = np.arange(start, int(np.floor(td)))
            xb = x0 + 1 - tb
            xc = x0 + 1 - tc
            weight = np.maximum(
                1 + (xb - np.abs(xb)) / (2 * (tb - ta))
                - (xc + np.abs(xc)) / (2 * (td - tc)),
                0,
            )
            u[x0] += lfilter(ar[i, :], [1.0], s[x0] - dc[i]) * weight

        x0 = np.arange(int(np.floor(tb)))
        xc = x0 + 1 - ta
        weight = 1 - (xc + np.abs(xc)) / (2 * (tb - ta))
        u[x0] += lfilter(ar[0, :], [1.0], s[x0] - dc[0]) * weight

    return u
