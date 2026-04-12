"""V_SPGRAMBW - Spectrogram computation with configurable bandwidth."""

from __future__ import annotations
import numpy as np
from .v_enframe import v_enframe
from .v_rfft import v_rfft
from .v_frq2mel import v_frq2mel
from .v_mel2frq import v_mel2frq
from .v_frq2bark import v_frq2bark
from .v_bark2frq import v_bark2frq
from .v_frq2erb import v_frq2erb
from .v_erb2frq import v_erb2frq


def v_spgrambw(s, fs, mode='', bw=200, fmax=None, db=40, tinc=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram with configurable bandwidth (no plotting).

    Parameters
    ----------
    s : array_like
        Speech signal or power spectrum array.
    fs : float or array_like
        Sample frequency in Hz, or [fs, t1].
    mode : str
        Mode options:
          'p' : output power per decade
          'P' : output power per mel/bark/erb
          'd' : output in dB
          'm' : mel scale
          'b' : bark scale
          'e' : erb scale
          'l' : log10 Hz scale
    bw : float
        Bandwidth resolution in Hz.
    fmax : array_like, optional
        Frequency range [Fmin, Fstep, Fmax].
    db : float
        dB range for plotting/clipping.
    tinc : float
        Output frame increment in seconds.

    Returns
    -------
    t : ndarray
        Time axis values (seconds).
    f : ndarray
        Frequency axis values.
    b : ndarray
        Spectrogram values (power per Hz unless mode changes this).
    """
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s[:, np.newaxis]
        ns1, ns2 = s.shape
        s = s.ravel()
        ns2 = 1
    else:
        ns1, ns2 = s.shape

    fs_arr = np.atleast_1d(np.asarray(fs, dtype=float))
    if len(fs_arr) < 2:
        fs_arr = np.array([fs_arr[0], 1.0 / fs_arr[0]])

    if tinc == 0:
        tinc = 1.81 / (4 * bw)

    # Determine frequency scale
    mdsw = ' '
    for ch in mode:
        if ch in 'lmbe':
            mdsw = ch

    nfrq = 257
    if ns2 == 1:
        fnyq = fs_arr[0] / 2.0
    else:
        if len(fs_arr) < 3:
            fs_arr = np.append(fs_arr, fs_arr[0] * 0.25)
        if len(fs_arr) < 4:
            fs_arr = np.append(fs_arr, 0)
        fnyq = fs_arr[3] + (ns2 - 1) * fs_arr[2]

    flmin = 30.0
    if fmax is None:
        if mdsw == 'l':
            fx = np.linspace(np.log10(flmin), np.log10(fnyq), nfrq)
        elif mdsw == 'm':
            fx = np.linspace(0, v_frq2mel(fnyq)[0], nfrq)
        elif mdsw == 'b':
            fx = np.linspace(0, v_frq2bark(fnyq)[0], nfrq)
        elif mdsw == 'e':
            fx = np.linspace(0, v_frq2erb(fnyq)[0], nfrq)
        else:
            fx = np.arange(nfrq) * fnyq / (nfrq - 1)
    else:
        fmax = np.atleast_1d(np.asarray(fmax, dtype=float))
        fmaxu = fmax.copy()
        if 'h' in mode:
            if mdsw == 'l':
                fmaxu = np.log10(fmax)
            elif mdsw == 'm':
                fmaxu = v_frq2mel(fmax)[0]
            elif mdsw == 'b':
                fmaxu = v_frq2bark(fmax)[0]
            elif mdsw == 'e':
                fmaxu = v_frq2erb(fmax)[0]

        if len(fmaxu) < 2:
            if mdsw == 'l':
                fx = np.linspace(np.log10(flmin), fmaxu[0], nfrq)
            else:
                fx = np.linspace(0, fmaxu[0], nfrq)
        elif len(fmaxu) < 3:
            fx = np.linspace(fmaxu[0], fmaxu[1], nfrq)
        else:
            fx = np.arange(fmaxu[0], fmaxu[2] + fmaxu[1] / 2, fmaxu[1])
            nfrq = len(fx)

    # Convert frequency range to Hz
    if mdsw == 'l':
        f_hz = 10.0 ** fx
    elif mdsw == 'm':
        f_hz = v_mel2frq(fx)[0]
    elif mdsw == 'b':
        f_hz = v_bark2frq(fx)[0]
    elif mdsw == 'e':
        f_hz = v_erb2frq(fx)[0]
    else:
        f_hz = fx.copy()

    f = fx.copy()  # output frequencies in native units

    # Calculate spectrogram
    if ns2 == 1:
        winlen = int(round(1.81 * fs_arr[0] / bw))
        win = 0.54 + 0.46 * np.cos(np.arange(1 - winlen, winlen + 1, 2) * np.pi / winlen)
        ninc = max(round(tinc * fs_arr[0]), 1)
        fftlen = int(2 ** np.ceil(np.log2(4 * winlen)))
        win = win / np.sqrt(np.sum(win ** 2))

        sf, t, *_ = v_enframe(s, win, ninc)
        t = fs_arr[1] + (t - 1) / fs_arr[0]  # time axis
        b_spec = v_rfft(sf, fftlen, 1)
        b_spec = np.real(b_spec * np.conj(b_spec)) * 2.0 / fs_arr[0]
        b_spec[:, 0] *= 0.5
        b_spec[:, -1] *= 0.5
        fb = np.arange(fftlen // 2 + 1) * fs_arr[0] / fftlen
    else:
        b_spec = s.copy()
        t = fs_arr[1] + np.arange(ns1) / fs_arr[0]
        fb = fs_arr[3] + np.arange(ns2) * fs_arr[2]
        fftlen = ns2

    nfr = len(t)

    # Apply preemphasis
    preemph = 'P' in mode
    if 'p' in mode or (preemph and mdsw == 'l'):
        b_spec = b_spec * fb[np.newaxis, :] * np.log(10)
    elif preemph and mdsw == 'm':
        b_spec = b_spec * ((700 + fb) * np.log(1 + 1000 / 700) / 1000)[np.newaxis, :]
    elif preemph and mdsw == 'b':
        b_spec = b_spec * ((1960 + fb) ** 2 / 52547.6)[np.newaxis, :]
    elif preemph and mdsw == 'e':
        b_spec = b_spec * (6.23 * fb ** 2 + 93.39 * fb + 28.52)[np.newaxis, :]

    # Interpolate onto desired frequency axis
    b_out = np.zeros((nfr, nfrq))
    for i in range(nfrq):
        # Find interpolation position
        fi = f_hz[i]
        idx = np.searchsorted(fb, fi)
        if idx <= 0:
            b_out[:, i] = b_spec[:, 0]
        elif idx >= len(fb):
            b_out[:, i] = b_spec[:, -1]
        else:
            frac = (fi - fb[idx - 1]) / (fb[idx] - fb[idx - 1])
            b_out[:, i] = (1 - frac) * b_spec[:, idx - 1] + frac * b_spec[:, idx]

    if 'd' in mode:
        b_out = 10.0 * np.log10(np.maximum(b_out, np.max(b_out) * 1e-30))

    return t, f, b_out
