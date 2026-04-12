"""V_ESTNOISEG - Estimate MMSE noise spectrum (Gerkmann & Hendriks)."""

from __future__ import annotations
import numpy as np


def v_estnoiseg(yf, tz, pp=None) -> tuple[np.ndarray, dict]:
    """Estimate noise spectrum using MMSE method.

    Parameters
    ----------
    yf : ndarray
        Input power spectra (one row per frame).
    tz : float or dict
        Frame increment in seconds, or state from previous call.
    pp : dict, optional
        Algorithm parameters.

    Returns
    -------
    x : ndarray
        Estimated noise power spectra (one row per frame).
    zo : dict
        Output state for subsequent calls.
    """
    yf = np.asarray(yf, dtype=float)
    if yf.ndim == 1:
        yf = yf.reshape(1, -1)
    nr, nrf = yf.shape
    x = np.zeros((nr, nrf))

    if nr == 0 and isinstance(tz, dict):
        return x, tz

    if isinstance(tz, dict):
        nrcum = tz['nrcum']
        xt = tz['xt']
        pslp = tz['pslp']
        tinc = tz['tinc']
        qq = tz['qq']
    else:
        tinc = tz
        nrcum = 0
        qq = {
            'tax': 0.0717,
            'tap': 0.152,
            'psthr': 0.99,
            'pnsaf': 0.01,
            'pspri': 0.5,
            'asnr': 15,
            'psini': 0.5,
            'tavini': 0.064,
        }
        if pp is not None:
            for key in qq:
                if key in pp:
                    qq[key] = pp[key]
        pslp = np.full(nrf, qq['psini'])
        xt = None

    # Unpack parameters
    psthr = qq['psthr']
    pnsaf = qq['pnsaf']

    # Derived constants
    ax = np.exp(-tinc / qq['tax'])
    axc = 1.0 - ax
    ap = np.exp(-tinc / qq['tap'])
    apc = 1.0 - ap
    xih1 = 10.0 ** (qq['asnr'] / 10.0)
    xih1r = 1.0 / (1.0 + xih1) - 1.0
    pfac = (1.0 / qq['pspri'] - 1.0) * (1.0 + xih1)

    if nrcum == 0 and nr > 0:
        nini = max(1, min(nr, round(1 + qq['tavini'] / tinc)))
        xt = qq['psini'] * np.mean(yf[:nini, :], axis=0)

    for t in range(nr):
        yft = yf[t, :]
        ph1y = (1.0 + pfac * np.exp(xih1r * yft / xt)) ** (-1)
        pslp = ap * pslp + apc * ph1y
        ph1y = np.minimum(ph1y, 1.0 - pnsaf * (pslp > psthr))
        xtr = (1.0 - ph1y) * yft + ph1y * xt
        xt = ax * xt + axc * xtr
        x[t, :] = xt

    zo = {
        'nrcum': nrcum + nr,
        'xt': xt,
        'pslp': pslp,
        'tinc': tinc,
        'qq': qq,
    }
    return x, zo
