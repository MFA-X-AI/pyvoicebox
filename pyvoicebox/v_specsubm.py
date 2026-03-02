"""V_SPECSUBM - Spectral subtraction (Martin's method)."""

import numpy as np
from .v_rfft import v_rfft
from .v_irfft import v_irfft


def v_specsubm(s, fs, p=None):
    """Spectral subtraction using minimum statistics (Martin).

    Parameters
    ----------
    s : array_like
        Input speech signal.
    fs : float
        Sample frequency in Hz.
    p : array_like, optional
        Algorithm parameters (11 elements).

    Returns
    -------
    ss : ndarray
        Enhanced speech output.
    po : ndarray
        Parameters used.
    """
    s = np.asarray(s, dtype=float).ravel()
    if p is None:
        po = np.array([0.04, 0.1, 0.032, 1.5, 0.08, 400, 4, 4, 1.5, 0.02, 4])
    else:
        po = np.asarray(p, dtype=float).ravel()

    ns = len(s)
    ts = 1.0 / fs
    ss = np.zeros(ns)

    ni = int(2 ** np.ceil(np.log2(fs * po[2] / po[7])))
    ti = ni / fs
    nw = int(ni * po[7])
    nf_count = 1 + int(np.floor((ns - nw) / ni))
    nm_count = int(np.ceil(fs * po[3] / (ni * po[6])))

    win = 0.5 * np.hamming(nw + 1)[:nw] / 1.08
    zg = np.exp(-ti / po[0])
    za = np.exp(-ti / po[1])
    zo = np.exp(-ti / po[4])

    px = np.zeros(1 + nw // 2)
    pxn = px.copy()
    os_val = px.copy()
    mb = np.ones((1 + nw // 2, int(po[6]))) * nw / 2
    im = 0
    osf = po[10] * (1.0 + np.arange(1 + nw // 2) * fs / (nw * po[5])) ** (-1)

    for i_s in range(nf_count):
        idx = np.arange(nw) + i_s * ni
        x = v_rfft(s[idx] * win)
        x2 = np.real(x * np.conj(x))

        pxn = za * pxn + (1.0 - za) * x2
        im = (im + 1) % nm_count
        if im:
            mb[:, 0] = np.minimum(mb[:, 0], pxn)
        else:
            mb = np.column_stack([pxn, mb[:, :int(po[6]) - 1]])
        pn = po[8] * np.min(mb, axis=1)

        os_val = zo * os_val + (1.0 - zo) * (1.0 + osf * pn / (pn + pxn))

        px = zg * px + (1.0 - zg) * x2
        q = np.maximum(po[9] * np.sqrt(pn / x2), 1.0 - np.sqrt(os_val * pn / px))
        ss[idx] += np.real(v_irfft(x * q, nw))

    return ss, po
