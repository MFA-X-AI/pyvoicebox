"""V_SSUBMMSE - Speech enhancement using MMSE spectral amplitude estimator."""

from __future__ import annotations
import numpy as np
from scipy.special import iv as besseli
from .v_enframe import v_enframe
from .v_rfft import v_rfft
from .v_irfft import v_irfft
from .v_estnoisem import v_estnoisem
from .v_estnoiseg import v_estnoiseg


def _expint(x):
    """Evaluate E1(x) = integral from x to inf of exp(-t)/t dt."""
    from scipy.special import exp1
    return exp1(x)


def v_ssubmmse(si, fs, pp=None) -> np.ndarray:
    """Speech enhancement using MMSE spectral amplitude estimator.

    Parameters
    ----------
    si : array_like
        Input speech signal (1-D vector).
    fs : float
        Sample frequency in Hz.
    pp : dict, optional
        Algorithm parameters.

    Returns
    -------
    ss : ndarray
        Enhanced speech output.
    """
    kk = np.sqrt(2 * np.pi)
    cc = np.sqrt(2.0 / np.pi)

    s = np.asarray(si, dtype=float).ravel()

    # Default parameters
    qq = {
        'of': 2, 'ti': 16e-3, 'ri': 0, 'ta': 0.396,
        'gx': 1000, 'gn': 1, 'gz': 0.001, 'xn': 0, 'xb': 1,
        'lg': 1, 'ne': 1, 'bt': -1, 'mx': 0, 'gc': 10,
        'tf': 'g', 'rf': 0,
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
    a = np.exp(-tinc / qq['ta'])
    gx = qq['gx']
    gz = qq['gz']
    xn = qq['xn']
    ne = qq['ne']
    gn1 = max(qq['gn'] - 1, 0)
    xb = qq['xb']

    # Calculate power spectrum
    no = round(qq['of'])
    nf = ni * no
    w = np.sqrt(np.hamming(nf + 1)[:nf])
    w = w / np.sqrt(np.sum(w[::ni] ** 2))

    y, tt, *_ = v_enframe(s, w, ni, 'r')
    tt = tt / fs
    yf = v_rfft(y, nf, 1)
    yp = np.real(yf * np.conj(yf))
    nr, nf2 = yp.shape

    if ne > 0:
        dp, ze = v_estnoiseg(yp, tinc, qp)
    else:
        dp, ze, _ = v_estnoisem(yp, tinc, qp)

    xu = 1.0

    if nr == 0:
        return np.array([])

    gam = np.clip(yp / dp, gz, gx)
    g = np.zeros((nr, nf2))
    x_snr = np.zeros((nr, nf2))

    if qq['lg'] == 0:  # amplitude domain
        for i in range(nr):
            gami = gam[i, :]
            xi = np.maximum(a * xb * xu + (1 - a) * np.maximum(gami - 1, gn1), xn)
            v = 0.5 * xi * gami / (1 + xi)
            gi = (0.277 + 2 * v) / gami
            mv = v < 0.5
            if np.any(mv):
                vmv = v[mv]
                gi[mv] = kk * np.sqrt(vmv) * ((0.5 + vmv) * besseli(0, vmv) + vmv * besseli(1, vmv)) / (gami[mv] * np.exp(vmv))
            g[i, :] = gi
            x_snr[i, :] = xi
            xu = gami * gi ** 2
    elif qq['lg'] == 2:  # perceptual
        for i in range(nr):
            gami = gam[i, :]
            xi = np.maximum(a * xb * xu + (1 - a) * np.maximum(gami - 1, gn1), xn)
            v = 0.5 * xi * gami / (1 + xi)
            gi = cc * np.sqrt(v) * np.exp(v) / (gami * besseli(0, v))
            g[i, :] = gi
            x_snr[i, :] = xi
            xu = gami * gi ** 2
    else:  # log domain (default)
        for i in range(nr):
            gami = gam[i, :]
            xi = np.maximum(a * xb * xu + (1 - a) * np.maximum(gami - 1, gn1), xn)
            xir = xi / (1 + xi)
            gi = xir * np.exp(0.5 * _expint(xir * gami))
            g[i, :] = gi
            x_snr[i, :] = xi
            xu = gami * gi ** 2

    g = np.minimum(qq['mx'] + (1 - qq['mx']) * g, qq['gc'])
    if qq['bt'] >= 0:
        g = (g > qq['bt']).astype(float)

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
