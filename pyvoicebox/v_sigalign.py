"""V_SIGALIGN - Align a clean reference with a noisy signal."""

from __future__ import annotations
import numpy as np
from pyvoicebox.v_rfft import v_rfft
from pyvoicebox.v_irfft import v_irfft


def v_sigalign(s, r, maxd=None, m='gs', fs=None) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Align a clean reference with a noisy signal.

    Parameters
    ----------
    s : array_like
        Test signal.
    r : array_like
        Reference signal.
    maxd : float or array_like, optional
        [+-max] or [min, max] delay allowed in samples. Fractions of len(r)
        are used if abs(maxd) < 1. Default ensures at least 50% overlap.
    m : str, optional
        Mode string:
          'u' - unity gain
          'g' - find optimal gain (default)
          's' - maximize correlation coefficient (default)
          'S' - maximize energy of common component
    fs : float, optional
        Sample frequency (only used for filtering).

    Returns
    -------
    d : int
        Optimum delay to apply to r.
    g : float
        Optimal gain to apply to r.
    rr : ndarray
        g * r(shifted by -d), zero-padded to match s if ss not returned.
    ss : ndarray
        s truncated to match rr.
    """
    s = np.asarray(s, dtype=float).ravel()
    r = np.asarray(r, dtype=float).ravel()
    ns = len(s)
    nr = len(r)

    if maxd is None:
        maxd_arr = np.array([])
    else:
        maxd_arr = np.atleast_1d(np.asarray(maxd, dtype=float))

    if len(maxd_arr) == 0:
        if nr < ns:
            lmm = np.array([-0.25 * nr, ns - 0.75 * nr])
        else:
            lmm = np.array([-0.25 * ns, nr - 0.75 * ns])
    elif len(maxd_arr) == 1:
        lmm = np.array([-maxd_arr[0], maxd_arr[0]])
    else:
        lmm = maxd_arr[:2].copy()

    # Convert fractions of nr to samples
    lmm = np.round(lmm * (1 + (nr - 1) * (np.abs(lmm) < 1))).astype(int)
    lmin = int(lmm[0])
    lmax = int(lmm[1])
    lags = lmax - lmin + 1

    if lags <= 0:
        raise ValueError('Invalid lag limits')

    # Note: A-weighting and BS-468 weighting (m='a', m='b') require
    # v_stdspectrum which is not yet converted. Skip filtering.

    # Cross correlation
    rxi = max(1, 1 - lmin)       # first reference sample needed (1-based)
    rxj = min(nr, ns - lmax)     # last reference sample needed (1-based)
    nrx = rxj - rxi + 1         # length of reference segment

    if nrx < 1:
        raise ValueError('Reference signal too short')

    fl = int(2 ** np.ceil(np.log2(lmax - lmin + nrx)))
    sxi = max(1, rxi + lmin)    # first signal sample needed (1-based)
    sxj = min(ns, rxj + lmax)   # last signal sample needed (1-based)

    # Zero-padded FFT cross-correlation
    s_seg = np.zeros(fl)
    s_seg[:sxj - sxi + 1] = s[sxi - 1:sxj]
    r_seg = np.zeros(fl)
    r_seg[:rxj - rxi + 1] = r[rxi - 1:rxj]

    S = np.fft.fft(s_seg)
    R = np.fft.fft(r_seg)
    rs_full = np.real(np.fft.ifft(S * np.conj(R)))
    rsu = rs_full[:lags]

    ssq = np.cumsum(s[sxi - 1:sxj] ** 2)
    ssqd = np.zeros(lags)
    ssqd[0] = ssq[nrx - 1]
    if lags > 1:
        ssqd[1:] = ssq[nrx:nrx + lags - 1] - ssq[:lags - 1]

    if 'S' in m:
        icx = np.argmax(np.abs(rsu))
    else:
        # Maximize correlation coefficient
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_coeff = rsu ** 2 / ssqd
            corr_coeff[ssqd == 0] = 0
        icx = np.argmax(corr_coeff)

    d = icx + lmin

    # Extract common region
    ia = max(1, d + 1)       # first sample of s in common region (1-based)
    ja = min(ns, d + nr)     # last sample of s in common region (1-based)
    ija = np.arange(ia, ja + 1)  # 1-based
    ijad = ija - d

    rr = r[ijad - 1].copy()
    ss = s[ija - 1].copy()

    if 'u' in m:
        g = 1.0
    else:
        g = np.sum(rr * ss) / np.sum(rr ** 2)

    rr = rr * g

    return d, g, rr, ss
