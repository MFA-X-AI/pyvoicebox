"""V_VADSOHN - Voice activity detector (Sohn et al.)."""

from __future__ import annotations
import numpy as np
from scipy.special import iv as besseli
from .v_enframe import v_enframe
from .v_rfft import v_rfft
from .v_estnoisem import v_estnoisem
from .v_estnoiseg import v_estnoiseg


def v_vadsohn(si, fs, m='a', pp=None) -> np.ndarray:
    """Voice activity detector based on Sohn et al.

    Parameters
    ----------
    si : array_like
        Input speech signal.
    fs : float
        Sample frequency in Hz.
    m : str
        Mode: 'a' activity decision, 'b' likelihood ratio.
    pp : dict, optional
        Algorithm parameters.

    Returns
    -------
    vs : ndarray
        VAD output (one value per sample if mode 'a', or per frame if 'n'/'t').
    """
    s = np.asarray(si, dtype=float).ravel()

    qq = {
        'of': 2, 'pr': 0.7, 'ts': 0.1, 'tn': 0.05,
        'ti': 10e-3, 'tj': 10e-3, 'ri': 0,
        'ta': 0.396, 'gx': 1000, 'gz': 1e-4, 'xn': 0, 'ne': 0,
    }
    qp = {}
    if pp is not None:
        for key in qq:
            if key in pp:
                qq[key] = pp[key]
        qp = pp

    qq['tj'] = min(qq['tj'], 0.5 * qq['ts'], 0.5 * qq['tn'])
    nj = max(round(qq['ti'] / qq['tj']), 1)
    if qq['ri']:
        ni = int(2 ** np.ceil(np.log2(qq['ti'] * fs * np.sqrt(0.5) / nj)))
    else:
        ni = round(qq['ti'] * fs / nj)

    tinc = ni / fs
    a_coeff = np.exp(-tinc / qq['ta'])
    gmax = qq['gx']
    kk = np.sqrt(2 * np.pi)
    xn = qq['xn']
    gz = qq['gz']
    a01 = tinc / qq['tn']
    a00 = 1 - a01
    a10 = tinc / qq['ts']
    a11 = 1 - a10
    b11 = a11 / a10
    b01 = a01 / a00
    b00 = a01 - a00 * a11 / a10
    b10 = a11 - a10 * a01 / a00
    prat = a10 / a01
    lprat = np.log(prat)

    no = round(qq['of'])
    nf = ni * no
    w = np.hamming(nf + 1)[:nf]
    w = w / np.sqrt(np.sum(w[::ni] ** 2))

    ns = len(s)
    y, *_ = v_enframe(s, w, ni)
    yf = v_rfft(y, nf, 1)

    if yf.shape[0] == 0:
        return np.array([])

    yp = np.real(yf * np.conj(yf))
    nr, nf2 = yp.shape
    nb = ni * nr

    if qq['ne'] > 0:
        dp, ze = v_estnoiseg(yp, tinc, qp)
    else:
        dp, ze, _ = v_estnoisem(yp, tinc, qp)

    xu = 1.0
    lggami = 0.0

    gam = np.clip(yp / dp, gz, gmax)
    prsp = np.zeros(nr)

    for i in range(nr):
        gami = gam[i, :]
        xi = a_coeff * xu + (1 - a_coeff) * np.maximum(gami - 1, xn)
        xi1 = 1 + xi
        v = 0.5 * xi * gami / xi1
        lamk = 2 * v - np.log(xi1)
        lami = np.sum(lamk[1:nf2]) / (nf2 - 1)

        if lggami < 0:
            lggami = lprat + lami + np.log(b11 + b00 / (a00 + a10 * np.exp(lggami)))
        else:
            lggami = lprat + lami + np.log(b01 + b10 / (a10 + a00 * np.exp(-lggami)))

        if lggami < 0:
            gg = np.exp(lggami)
            prsp[i] = gg / (1 + gg)
        else:
            prsp[i] = 1.0 / (1 + np.exp(-lggami))

        gi = (0.277 + 2 * v) / gami
        mv = v < 0.5
        if np.any(mv):
            vmv = v[mv]
            gi[mv] = kk * np.sqrt(vmv) * ((0.5 + vmv) * besseli(0, vmv) + vmv * besseli(1, vmv)) / (gami[mv] * np.exp(vmv))
        xu = gami * gi ** 2

    if 'a' in m:
        # Output per sample
        vs = np.zeros(ns)
        for i in range(nr):
            start = i * ni
            end = min(start + ni, ns)
            vs[start:end] = float(prsp[i] > qq['pr'])
        return vs
    elif 'b' in m:
        vs = np.zeros(ns)
        for i in range(nr):
            start = i * ni
            end = min(start + ni, ns)
            vs[start:end] = prsp[i]
        return vs
    else:
        vs = np.zeros(ns)
        for i in range(nr):
            start = i * ni
            end = min(start + ni, ns)
            vs[start:end] = float(prsp[i] > qq['pr'])
        return vs
