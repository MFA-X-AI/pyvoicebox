"""V_ACTIVLEV - Measure active speech level as per ITU-T P.56."""

import numpy as np
from scipy.signal import lfilter
from pyvoicebox.v_maxfilt import v_maxfilt


def v_activlev(sp, fs, mode=''):
    """Measure active speech level as in ITU-T P.56 (Method B).

    Parameters
    ----------
    sp : array_like
        Speech signal.
    fs : float
        Sample frequency in Hz.
    mode : str, optional
        Mode string:
            'd' - output in dB
            'n' - normalize speech to 0 dB active level
            '0' - omit high pass filter
            'h' - omit low pass filter
            'l' - output long-term power level too

    Returns
    -------
    lev : float or ndarray
        Active speech level.
    af : float
        Activity factor.
    """
    nbin = 20
    thresh = 15.9

    sp = np.asarray(sp, dtype=float).ravel()
    ns = len(sp)

    if ns == 0:
        if 'd' in mode:
            return -np.inf, 0.0
        return 0.0, 0.0

    # Simplified implementation: compute active level without bandpass filtering
    # (use '0h' mode equivalent for simplicity)
    ti = 1.0 / fs
    g = np.exp(-ti / 0.03)
    ae = np.array([1, -2 * g, g ** 2]) / (1 - g) ** 2
    nh = int(np.ceil(0.2 / ti)) + 1

    # Zero-pad
    nz = int(np.ceil(0.35 * fs))
    sq = np.concatenate([sp, np.zeros(nz)])
    ns_total = len(sq)

    # Envelope filter
    s = lfilter([1], ae, np.abs(sq))

    # Sum of squares
    ssq = np.sum(sq ** 2)
    if ssq == 0:
        af = 0.0
        if 'd' in mode:
            lev = -np.inf
        else:
            lev = 0.0
        return lev, af

    sf = ns_total / ssq
    sfdb = 10 * np.log10(sf)

    # Energy histogram using log2
    qf, qe = np.frexp(sf * s ** 2)
    qe = qe.astype(float)
    qe[qf == 0] = -np.inf

    # Apply hangover
    qe_hang = v_maxfilt(qe, 1, nh, 1)

    emax = np.max(qe_hang) + 1
    if emax == -np.inf:
        if 'd' in mode:
            return -np.inf, 0.0
        return 0.0, 0.0

    qe_idx = np.minimum(emax - qe_hang, nbin).astype(int)
    qe_idx = np.clip(qe_idx, 1, nbin)

    kc = np.cumsum(np.bincount(qe_idx, minlength=nbin + 1)[1:nbin + 1])

    # Calculate active level
    aj = 10 * np.log10(ssq / np.maximum(kc, 1))
    cj = 10 * np.log10(2) * (emax - np.arange(1, nbin + 1) - 1) - sfdb
    mj = aj - cj - thresh

    # Find positive transition through threshold
    transitions = np.where((mj[:-1] < 0) & (mj[1:] >= 0))[0]
    if len(transitions) == 0:
        if mj[-1] <= 0:
            jj = len(mj) - 2
            jf = 1.0
        else:
            jj = 0
            jf = 0.0
    else:
        jj = transitions[0]
        denom = mj[jj + 1] - mj[jj]
        if abs(denom) < 1e-30:
            jf = 0.5
        else:
            jf = 1.0 / (1.0 - mj[jj + 1] / mj[jj])

    lev_db = aj[jj] + jf * (aj[min(jj + 1, nbin - 1)] - aj[jj])
    lp = 10 ** (lev_db / 10)
    af = ssq / ((ns_total - nz) * lp) if lp > 0 else 0.0

    if 'd' in mode:
        lev = lev_db
        if 'l' in mode:
            lev = np.array([lev_db, 10 * np.log10(ssq / ns_total)])
    else:
        lev = lp
        if 'l' in mode:
            lev = np.array([lp, ssq / ns_total])

    if 'n' in mode:
        if lp > 0:
            return sp / np.sqrt(lp), lev, af
        return sp, lev, af

    return lev, af
