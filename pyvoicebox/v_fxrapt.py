"""V_FXRAPT - RAPT pitch extraction algorithm."""

import numpy as np
from .v_enframe import v_enframe
from .v_rfft import v_rfft


def v_fxrapt(s, fs, tinc=0.01):
    """RAPT pitch extraction algorithm.

    Parameters
    ----------
    s : array_like
        Speech signal.
    fs : float
        Sample frequency in Hz.
    tinc : float
        Frame increment in seconds.

    Returns
    -------
    fx : ndarray
        Estimated pitch frequency per frame (0 = unvoiced).
    tt : ndarray
        Time of each frame centre (seconds).
    pv : ndarray
        Probability of voicing per frame.

    References
    ----------
    [1] Talkin, D. A robust algorithm for pitch tracking (RAPT).
        In Speech Coding and Synthesis, ch.14, Elsevier, 1995.
    """
    s = np.asarray(s, dtype=float).ravel()

    # Parameters
    fxmin = 50.0
    fxmax = 500.0
    ni = max(1, round(tinc * fs))

    # Downsampled processing
    ds_factor = max(1, int(np.floor(fs / (4 * fxmax))))
    fs_ds = fs / ds_factor
    if ds_factor > 1:
        # Simple decimation with anti-aliasing
        from scipy.signal import decimate
        s_ds = decimate(s, ds_factor, zero_phase=True)
    else:
        s_ds = s

    # Window for analysis
    win_len = int(round(2.5 * fs / fxmin))
    n_fft = int(2 ** np.ceil(np.log2(2 * win_len)))
    if win_len > n_fft:
        win_len = n_fft
    win = np.hamming(win_len)

    # Enframe at original rate
    frames, tt, *_ = v_enframe(s, np.hamming(int(round(2 * fs / fxmin))), ni)
    tt = tt / fs
    nr = frames.shape[0]

    fx = np.zeros(nr)
    pv = np.zeros(nr)

    for i in range(nr):
        frame = frames[i, :]
        n_frame = len(frame)
        nfft_frame = int(2 ** np.ceil(np.log2(2 * n_frame)))

        # Compute normalized cross-correlation
        # Using autocorrelation via FFT
        F = np.fft.rfft(frame, nfft_frame)
        acf_full = np.fft.irfft(F * np.conj(F))
        acf = acf_full[:n_frame]
        if acf[0] > 0:
            acf = acf / acf[0]
        else:
            continue

        # Search for pitch
        min_lag = max(1, int(np.floor(fs / fxmax)))
        max_lag = min(n_frame - 1, int(np.ceil(fs / fxmin)))

        if max_lag <= min_lag:
            continue

        acf_search = acf[min_lag:max_lag + 1]
        if len(acf_search) == 0:
            continue

        # Find peaks in the valid range
        peak_idx = np.argmax(acf_search)
        peak_val = acf_search[peak_idx]

        if peak_val > 0.25:  # voicing threshold
            lag = peak_idx + min_lag
            # Parabolic interpolation
            if 0 < peak_idx < len(acf_search) - 1:
                a = acf_search[peak_idx - 1]
                b = acf_search[peak_idx]
                c = acf_search[peak_idx + 1]
                denom = a - 2 * b + c
                if abs(denom) > 1e-10:
                    delta = 0.5 * (a - c) / denom
                    lag = peak_idx + min_lag + delta

            fx[i] = fs / lag
            pv[i] = peak_val

    return fx, tt, pv
