"""V_SSUBMMSEV - Speech enhancement using MMSE with VAD-based noise estimation."""

from __future__ import annotations
import numpy as np
from scipy.special import iv as besseli
from .v_enframe import v_enframe
from .v_rfft import v_rfft
from .v_irfft import v_irfft


def _expint(x):
    """Evaluate E1(x) = integral from x to inf of exp(-t)/t dt."""
    from scipy.special import exp1
    return exp1(x)


def v_ssubmmsev(si, fs, pp=None) -> np.ndarray:
    """Speech enhancement using MMSE with VAD-based noise estimation.

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

    qq = {
        'of': 2, 'ti': 16e-3, 'ri': 0, 'ta': 0.396,
        'gx': 1000, 'gn': 1, 'gz': 0.001, 'xn': 0, 'xb': 1,
        'lg': 1, 'ne': 0, 'bt': -1, 'mx': 0,
        'tf': 'g', 'rf': 0,
        'tn': 0.5, 'le': 0.15, 'tx': 0.06,
    }
    if pp is not None:
        for key in qq:
            if key in pp:
                qq[key] = pp[key]

    if qq['ri']:
        ni = int(2 ** np.ceil(np.log2(qq['ti'] * fs * np.sqrt(0.5))))
    else:
        ni = round(qq['ti'] * fs)
    tinc = ni / fs
    a_coeff = np.exp(-tinc / qq['ta'])
    gx = qq['gx']
    gz = qq['gz']
    xn = qq['xn']
    gn1 = max(qq['gn'] - 1, 0)
    le = qq['le']
    xb = qq['xb']
    nd = max(1, round(qq['tx'] / tinc))
    an = np.exp(-tinc / qq['tn'])

    no = round(qq['of'])
    nf = ni * no
    w = np.sqrt(np.hamming(nf + 1)[:nf])
    w = w / np.sqrt(np.sum(w[::ni] ** 2))

    y, tt, *_ = v_enframe(s, w, ni, 'r')
    tt = tt / fs
    yf = v_rfft(y, nf, 1)
    yp = np.real(yf * np.conj(yf))
    nr, nf2 = yp.shape

    dpi = np.zeros(nf2)
    ndp = 0
    xu = 1.0

    if nr == 0:
        return np.array([])

    # Initialize noise estimate
    ndx = min(nr, nd - ndp)
    dpi = np.mean(yp[:ndx, :], axis=0)
    ndp = ndx

    g = np.zeros((nr, nf2))

    if qq['lg'] == 0:
        for i in range(nr):
            ypi = yp[i, :]
            gami = np.clip(ypi / dpi, gz, gx)
            xi = np.maximum(a_coeff * xb * xu + (1 - a_coeff) * np.maximum(gami - 1, gn1), xn)
            if np.sum(gami * xi / (1 + xi) - np.log(1 + xi)) < le * nf2:
                dpi = dpi * an + (1 - an) * ypi
            v = 0.5 * xi * gami / (1 + xi)
            gi = (0.277 + 2 * v) / gami
            mv = v < 0.5
            if np.any(mv):
                vmv = v[mv]
                gi[mv] = kk * np.sqrt(vmv) * ((0.5 + vmv) * besseli(0, vmv) + vmv * besseli(1, vmv)) / (gami[mv] * np.exp(vmv))
            g[i, :] = gi
            xu = gami * gi ** 2
    elif qq['lg'] == 2:
        for i in range(nr):
            ypi = yp[i, :]
            gami = np.clip(ypi / dpi, gz, gx)
            xi = np.maximum(a_coeff * xb * xu + (1 - a_coeff) * np.maximum(gami - 1, gn1), xn)
            if np.sum(gami * xi / (1 + xi) - np.log(1 + xi)) < le * nf2:
                dpi = dpi * an + (1 - an) * ypi
            v = 0.5 * xi * gami / (1 + xi)
            gi = cc * np.sqrt(v) * np.exp(v) / (gami * besseli(0, v))
            g[i, :] = gi
            xu = gami * gi ** 2
    else:
        for i in range(nr):
            ypi = yp[i, :]
            gami = np.clip(ypi / dpi, gz, gx)
            xi = np.maximum(a_coeff * xb * xu + (1 - a_coeff) * np.maximum(gami - 1, gn1), xn)
            xir = xi / (1 + xi)
            if np.sum(gami * xir - np.log(1 + xi)) < le * nf2:
                dpi = dpi * an + (1 - an) * ypi
            gi = xir * np.exp(0.5 * _expint(xir * gami))
            g[i, :] = gi
            xu = gami * gi ** 2

    if qq['bt'] >= 0:
        g = (g > qq['bt']).astype(float)
    g = qq['mx'] + (1 - qq['mx']) * g

    se = np.real(v_irfft((yf * g).T, nf).T) * w[np.newaxis, :]
    ss = np.zeros(ni * (nr + no - 1))
    for i in range(no):
        frames = np.arange(i, nr, no)
        nm = nf * len(frames)
        seg = se[frames, :].reshape(-1)
        ss[i * ni:i * ni + nm] += seg

    ss = ss[:len(s)]
    return ss
