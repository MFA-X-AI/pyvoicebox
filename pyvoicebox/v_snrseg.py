"""V_SNRSEG - Measure segmental and global SNR."""

from __future__ import annotations
import numpy as np


def v_snrseg(s, r, fs, m='wz', tf=0.01) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Measure segmental and global SNR.

    Parameters
    ----------
    s : array_like
        Test signal (noisy).
    r : array_like
        Reference signal (clean).
    fs : float
        Sample frequency in Hz.
    m : str, optional
        Mode string:
          'w' : No VAD - use whole file (default).
          'z' : Do not do any alignment (default).
          'q' : Use linear interpolation to remove delays +/- 1 sample.
    tf : float, optional
        Frame increment in seconds. Default: 0.01.

    Returns
    -------
    seg : float
        Segmental SNR in dB.
    glo : float
        Global SNR in dB.
    tc : ndarray
        Time at centre of each frame (seconds).
    snf : ndarray
        Segmental SNR in dB in each frame.
    vf : ndarray
        Boolean mask indicating valid frames.
    """
    s = np.asarray(s, dtype=float).ravel()
    r = np.asarray(r, dtype=float).ravel()
    snmax = 100.0  # clipping limit for SNR

    mq = 'z' not in m  # flag for alignment
    nr = min(len(r), len(s))
    kf = round(tf * fs)  # frame length in samples
    ifr_start = kf + mq  # starting sample for first frame end (1-based in MATLAB)

    # Build frame end indices (0-based)
    ifr = np.arange(ifr_start, nr - mq + 1, kf) - 1  # 0-based ending sample
    nf = len(ifr)
    if nf == 0:
        return 0.0, 0.0, np.array([]), np.array([]), np.array([])

    ifl = ifr[-1] + 1  # last sample index + 1 (exclusive)

    if mq:
        # Linear interpolation alignment
        # For each frame, find optimal shift
        # s[1:ifl-1] vs s[2:ifl], s[0:ifl-2] (0-based)
        ssm = np.zeros((kf, nf))
        ssp = np.zeros((kf, nf))
        sr_mat = np.zeros((kf, nf))
        for j in range(nf):
            start = j * kf + 1  # 1-based converted: frame starts at mq + j*kf (0-based)
            for k in range(kf):
                idx = start + k  # 0-based index into s[1:ifl]
                ssm[k, j] = s[idx] - s[idx + 1]
                ssp[k, j] = s[idx] - s[idx - 1]
                sr_mat[k, j] = s[idx] - r[idx]

        am = np.clip(np.sum(sr_mat * ssm, axis=0) / np.sum(ssm**2, axis=0), 0, 1)
        ap = np.clip(np.sum(sr_mat * ssp, axis=0) / np.sum(ssp**2, axis=0), 0, 1)
        ef = np.minimum(
            np.sum((sr_mat - am[np.newaxis, :] * ssm)**2, axis=0),
            np.sum((sr_mat - ap[np.newaxis, :] * ssp)**2, axis=0)
        )
    else:
        # No interpolation
        ef = np.zeros(nf)
        for j in range(nf):
            start = j * kf
            frame_s = s[start:start + kf]
            frame_r = r[start:start + kf]
            ef[j] = np.sum((frame_s - frame_r)**2)

    # Calculate reference power per frame
    rf = np.zeros(nf)
    for j in range(nf):
        start = mq + j * kf
        rf[j] = np.sum(r[start:start + kf]**2)

    em = ef == 0  # zero noise frames
    rm = rf == 0  # zero reference frames
    snf = 10.0 * np.log10((rf + rm) / (ef + em))
    snf[rm] = -snmax
    snf[em] = snmax

    # Select frames to include
    if 'w' in m:
        vf = np.ones(nf, dtype=bool)
    else:
        vf = np.ones(nf, dtype=bool)  # default: use all

    tc = (np.arange(1, nf + 1) * kf + (1 - kf) / 2.0) / fs

    seg = np.mean(snf[vf]) if np.any(vf) else 0.0
    glo = 10.0 * np.log10(np.sum(rf[vf]) / np.sum(ef[vf])) if np.any(vf) and np.sum(ef[vf]) > 0 else 0.0

    return seg, glo, tc, snf, vf
