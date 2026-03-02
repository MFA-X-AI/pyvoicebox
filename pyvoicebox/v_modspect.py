"""V_MODSPECT - Calculate modulation spectrum of a signal."""

import numpy as np
from .v_enframe import v_enframe
from .v_rfft import v_rfft
from .v_melbankm import v_melbankm


def v_modspect(s, fs=11025, m='', nf=None, nq=None):
    """Calculate the modulation spectrum of a signal.

    This is a simplified implementation that computes the mel spectrogram
    and then the modulation spectrum for each mel channel.

    Parameters
    ----------
    s : array_like
        Speech signal.
    fs : float
        Sample rate in Hz.
    m : str
        Mode string.
    nf : array_like, optional
        [num_mel_bins, fmin, fmax, num_dct].
    nq : array_like, optional
        [num_mod_bins, mod_fmin, mod_fmax, num_mod_dct].

    Returns
    -------
    c : ndarray
        Modulation spectrum (mod_freq, mel_freq, time).
    qq : ndarray
        Modulation frequency centres.
    ff : ndarray
        Mel frequency centres.
    tt : ndarray
        Time axis.
    """
    s = np.asarray(s, dtype=float).ravel()

    # Parse nf parameters
    nf_defaults = [0.1, 40, min(10000, fs / 2), 25]
    if nf is not None:
        nf = list(np.atleast_1d(nf))
        while len(nf) < 4:
            nf.append(nf_defaults[len(nf)])
    else:
        nf = nf_defaults

    nq_defaults = [0.1, 0.5, 20, 15]
    if nq is not None:
        nq = list(np.atleast_1d(nq))
        while len(nq) < 4:
            nq.append(nq_defaults[len(nq)])
    else:
        nq = nq_defaults

    # Step 1: Short-time Fourier Transform
    n_fft = int(2 ** np.floor(np.log2(0.03 * fs)))
    inc = n_fft // 2
    win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n_fft) / (n_fft - 1))
    frames, tt, *_ = v_enframe(s, win, inc)
    tt = tt / fs

    spec = v_rfft(frames, n_fft, 1)
    pw = np.real(spec * np.conj(spec))

    # Step 2: Mel filterbank
    n_mel = int(nf[0]) if nf[0] >= 1 else int(np.ceil(4.6 * np.log10(fs)))
    mb, mc, _, _ = v_melbankm(n_mel, n_fft, fs, nf[1] / fs, nf[2] / fs)
    mel_spec = np.sqrt(np.maximum(mb.toarray() @ pw.T, 1e-20))  # (n_mel, n_frames)

    n_frames = mel_spec.shape[1]
    ff = mc

    # Step 3: Modulation spectrum per mel channel
    mod_inc = max(1, round(0.01 * fs / inc))  # ~10ms in frame units
    mod_win_len = max(4, min(n_frames, round(0.2 * fs / inc)))
    mod_nfft = int(2 ** np.ceil(np.log2(mod_win_len)))

    n_mod_frames = max(1, (n_frames - mod_win_len) // mod_inc + 1)
    n_mod_bins = mod_nfft // 2 + 1

    mod_freqs = np.arange(n_mod_bins) / mod_nfft * (fs / inc)
    c = np.zeros((n_mod_bins, n_mel, n_mod_frames))

    mod_win = np.hamming(mod_win_len)
    for i_mel in range(n_mel):
        sig = mel_spec[i_mel, :]
        for i_frame in range(n_mod_frames):
            start = i_frame * mod_inc
            end = start + mod_win_len
            if end > n_frames:
                break
            seg = sig[start:end] * mod_win
            f_mod = np.fft.rfft(seg, mod_nfft)
            c[:, i_mel, i_frame] = np.abs(f_mod) ** 2

    qq = mod_freqs
    tt_out = tt[::mod_inc][:n_mod_frames] if len(tt) > 0 else np.arange(n_mod_frames)

    return c, qq, ff, tt_out
