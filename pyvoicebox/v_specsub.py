"""V_SPECSUB - Speech enhancement using spectral subtraction."""

import numpy as np
from .v_enframe import v_enframe
from .v_rfft import v_rfft
from .v_irfft import v_irfft
from .v_estnoisem import v_estnoisem
from .v_estnoiseg import v_estnoiseg


def v_specsub(si, fs, pp=None):
    """Perform speech enhancement using spectral subtraction.

    Parameters
    ----------
    si : array_like
        Input speech signal.
    fs : float
        Sample frequency in Hz.
    pp : dict, optional
        Algorithm parameters.

    Returns
    -------
    ss : ndarray
        Enhanced speech output.
    """
    s = np.asarray(si, dtype=float).ravel()

    # Default parameters
    qq = {
        'of': 2, 'ti': 16e-3, 'ri': 0, 'g': 1, 'e': 1,
        'am': 3, 'b': 0.01, 'al': -5, 'ah': 20,
        'bt': -1, 'ne': 0, 'mx': 0, 'gh': 1, 'tf': 'g', 'rf': 0,
    }
    qp = {}
    if pp is not None:
        for key in qq:
            if key in pp:
                qq[key] = pp[key]
        qp = pp

    # Derived constants
    if qq['ri']:
        ni = int(2 ** np.ceil(np.log2(qq['ti'] * fs * np.sqrt(0.5))))
    else:
        ni = round(qq['ti'] * fs)
    tinc = ni / fs
    ne = qq['ne']

    # Calculate power spectrum
    no = round(qq['of'])
    nf = ni * no
    w = np.sqrt(np.hamming(nf + 1)[:nf])
    w = w / np.sqrt(np.sum(w[::ni] ** 2))

    y, tt, *_ = v_enframe(s, w, ni, 'r')
    tt = tt / fs
    yf = v_rfft(y, nf, 1)  # axis=1 for row-wise FFT
    yp = np.real(yf * np.conj(yf))
    nr, nf2 = yp.shape

    ff = np.arange(nf2) * fs / nf

    if ne > 0:
        dp, ze = v_estnoiseg(yp, tinc, qp)
    else:
        dp, ze, _ = v_estnoisem(yp, tinc, qp)

    ssv = np.zeros(ni * (no - 1))

    if nr == 0:
        return np.array([])

    mz = yp == 0
    if qq['al'] < np.inf:
        ypf = np.sum(yp, axis=1)
        dpf = np.sum(dp, axis=1)
        mzf = dpf == 0
        af = 1.0 + (qq['am'] - 1.0) * (np.clip(10.0 * np.log10(ypf / (dpf + mzf)), qq['al'], qq['ah']) - qq['ah']) / (qq['al'] - qq['ah'])
        af[mzf] = 1.0
    else:
        af = np.full(nr, qq['am'])

    if qq['g'] == 1:
        v = np.sqrt(dp / (yp + mz))
        af = np.sqrt(af)
        bf = np.sqrt(qq['b'])
    elif qq['g'] == 2:
        v = dp / (yp + mz)
        bf = qq['b']
    else:
        v = (dp / (yp + mz)) ** (0.5 * qq['g'])
        af = af ** (0.5 * qq['g'])
        bf = qq['b'] ** (0.5 * qq['g'])

    af = af[:, np.newaxis] * np.ones((1, nf2))
    mf = v >= 1.0 / (af + bf)
    g = np.zeros_like(v)
    eg = qq['e'] / qq['g']
    gh = qq['gh']

    if eg == 1:
        g[mf] = np.minimum(bf * v[mf], gh)
        g[~mf] = 1.0 - af[~mf] * v[~mf]
    elif eg == 0.5:
        g[mf] = np.minimum(np.sqrt(bf * v[mf]), gh)
        g[~mf] = np.sqrt(1.0 - af[~mf] * v[~mf])
    else:
        g[mf] = np.minimum((bf * v[mf]) ** eg, gh)
        g[~mf] = (1.0 - af[~mf] * v[~mf]) ** eg

    if qq['bt'] >= 0:
        g = (g > qq['bt']).astype(float)

    g = qq['mx'] + (1.0 - qq['mx']) * g

    # Inverse FFT and overlap-add
    se = np.real(v_irfft((yf * g).T, nf).T) * w[np.newaxis, :]
    ss = np.zeros(ni * (nr + no - 1))
    for i in range(no):
        frames = np.arange(i, nr, no)
        nm = nf * len(frames)
        seg = se[frames, :].reshape(-1)
        ss[i * ni:i * ni + nm] += seg

    ss = ss[:len(s)]
    return ss
