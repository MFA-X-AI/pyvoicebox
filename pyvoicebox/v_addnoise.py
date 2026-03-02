"""V_ADDNOISE - Add noise at a chosen SNR."""

import numpy as np


def v_addnoise(s, fs, snr=np.inf, m=''):
    """Add white noise at a chosen SNR using energy-based measurement.

    This is a simplified version that supports white noise addition with
    energy-based level measurement. For more advanced noise types, see the
    MATLAB original.

    Parameters
    ----------
    s : array_like
        Input speech signal (1-D).
    fs : float
        Sample frequency in Hz.
    snr : float, optional
        Target SNR in dB. Default: Inf (no noise).
    m : str, optional
        Mode string:
          'D' : SNR input given as power ratio instead of dB.
          'e' : Use energy to calculate signal level (default).
          'k' : Preserve original signal power.
          'n' : Make signal level = 0 dB.
          'N' : Make noise level = 0 dB.
          't' : Make total = 0 dB (default).
          'x' : Output signal and noise as separate columns.

    Returns
    -------
    z : ndarray
        Noisy signal (or [signal, noise] if 'x' option).
    p : ndarray
        Levels: [s-in n-in s-out n-out] as power ratios or dB.
    """
    s = np.asarray(s, dtype=float).ravel()
    ns = len(s)

    # Calculate signal energy
    se = np.mean(s**2)

    # Generate white noise
    n = np.random.randn(ns)
    ne = np.mean(n**2)

    # Convert SNR
    if 'D' in m:
        snre = snr
    else:
        snre = 10.0 ** (0.1 * snr)

    # Determine scaling factors
    if snre > 1:
        if 'n' in m:
            sze = 1.0
        elif 'N' in m:
            sze = snre
        elif 'k' in m:
            sze = se
        else:
            sze = 1.0 / (1.0 + 1.0 / snre)
        nze = sze / snre
    else:
        if 'n' in m:
            nze = 1.0 / snre
        elif 'N' in m:
            nze = 1.0
        elif 'k' in m:
            nze = se / snre
        else:
            nze = 1.0 / (1.0 + snre)
        sze = nze * snre

    pe = np.array([se, ne, sze, nze])
    gm = np.array([
        np.sqrt(sze / (se + (se == 0))),
        np.sqrt(nze / (ne + (ne == 0)))
    ])

    if gm[1] > 0:
        if 'x' in m:
            z = np.column_stack([gm[0] * s, gm[1] * n])
        else:
            z = gm[0] * s + gm[1] * n
    elif 'x' in m:
        z = np.column_stack([gm[0] * s, np.zeros(ns)])
    else:
        z = gm[0] * s

    p = pe.copy()
    if 'D' not in m:
        mk = (pe != np.inf) & (pe != 0)
        p[mk] = 10.0 * np.log10(pe[mk])
        p[pe == 0] = -np.inf

    return z, p
