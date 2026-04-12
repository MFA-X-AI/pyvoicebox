"""V_ESTNOISEM - Estimate noise spectrum using minimum statistics (Martin)."""

from __future__ import annotations
import numpy as np


def _mhvals(d):
    """Calculate M(D) and H(D) from Table 5 of Martin 2006."""
    dmh = np.array([
        [1, 0, 0],
        [2, 0.26, 0.15],
        [5, 0.48, 0.48],
        [8, 0.58, 0.78],
        [10, 0.61, 0.98],
        [15, 0.668, 1.55],
        [20, 0.705, 2],
        [30, 0.762, 2.3],
        [40, 0.8, 2.52],
        [60, 0.841, 3.1],
        [80, 0.865, 3.38],
        [120, 0.89, 4.15],
        [140, 0.9, 4.35],
        [160, 0.91, 4.25],
        [180, 0.92, 3.9],
        [220, 0.93, 4.1],
        [260, 0.935, 4.7],
        [300, 0.94, 5],
    ])

    # Find first index where dmh[:,0] >= d
    ii = np.where(d <= dmh[:, 0])[0]
    if len(ii) == 0:
        i = len(dmh) - 1
        j = i
    else:
        i = ii[0]
        j = max(i - 1, 0)

    if d == dmh[i, 0]:
        m = dmh[i, 1]
        h = dmh[i, 2]
    else:
        qj = np.sqrt(dmh[j, 0])
        qi = np.sqrt(dmh[i, 0])
        q = np.sqrt(d)
        h = dmh[i, 2] + (q - qi) * (dmh[j, 2] - dmh[i, 2]) / (qj - qi)
        m = dmh[i, 1] + (qi * qj / q - qj) * (dmh[j, 1] - dmh[i, 1]) / (qi - qj)

    return m, h


def v_estnoisem(yf, tz, pp=None) -> tuple[np.ndarray, dict, np.ndarray]:
    """Estimate noise spectrum using minimum statistics.

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
        Output state.
    xs : ndarray
        Estimated std error of x.
    """
    yf = np.asarray(yf, dtype=float)
    if yf.ndim == 1:
        yf = yf.reshape(1, -1)
    nr, nrf = yf.shape
    x = np.zeros((nr, nrf))
    xs = np.zeros((nr, nrf))

    if nr == 0 and isinstance(tz, dict):
        return x, tz, xs

    if isinstance(tz, dict):
        nrcum = tz['nrcum']
        p_state = tz['p']
        ac = tz['ac']
        sn2 = tz['sn2']
        pb = tz['pb']
        pb2 = tz['pb2']
        pminu = tz['pminu']
        actmin = tz['actmin']
        actminsub = tz['actminsub']
        subwc = tz['subwc']
        actbuf = tz['actbuf']
        ibuf = tz['ibuf']
        lminflag = tz['lminflag']
        tinc = tz['tinc']
        qq = tz['qq']
    else:
        tinc = tz
        nrcum = 0
        qq = {
            'taca': 0.0449,
            'tamax': 0.392,
            'taminh': 0.0133,
            'tpfall': 0.064,
            'tbmax': 0.0717,
            'qeqmin': 2,
            'qeqmax': 14,
            'av': 2.12,
            'td': 1.536,
            'nu': 8,
            'qith': np.array([0.03, 0.05, 0.06, np.inf]),
            'nsmdb': np.array([47, 31.4, 15.7, 4.1]),
        }
        if pp is not None:
            for key in qq:
                if key in pp:
                    qq[key] = pp[key]

    # Unpack parameters
    taca = qq['taca']
    tamax = qq['tamax']
    taminh = qq['taminh']
    tpfall = qq['tpfall']
    tbmax = qq['tbmax']
    qeqmin = qq['qeqmin']
    qeqmax = qq['qeqmax']
    av = qq['av']
    td = qq['td']
    nu = qq['nu']
    qith = np.asarray(qq['qith'])
    nsmdb = np.asarray(qq['nsmdb'])

    # Derived constants
    aca = np.exp(-tinc / taca)
    acmax = aca
    amax = np.exp(-tinc / tamax)
    aminh = np.exp(-tinc / taminh)
    bmax = np.exp(-tinc / tbmax)
    snrexp = -tinc / tpfall
    nv = round(td / (tinc * nu))
    if nv < 4:
        nv = 4
        nu = max(round(td / (tinc * nv)), 1)
    nd = nu * nv

    md, hd = _mhvals(nd)
    mv, hv = _mhvals(nv)
    nsms = 10.0 ** (nsmdb * nv * tinc / 10.0)
    qeqimax = 1.0 / qeqmin
    qeqimin = 1.0 / qeqmax

    if not isinstance(tz, dict):
        if nr == 0:
            ac = 1.0
            subwc = nv
            ibuf = 0
            p_state = x.copy()
            sn2 = p_state.copy()
            pb = p_state.copy()
            pb2 = pb ** 2
            pminu = p_state.copy()
            actmin = np.full(nrf, np.inf)
            actminsub = actmin.copy()
            actbuf = np.full((nu, nrf), np.inf)
            lminflag = np.zeros(nrf, dtype=bool)
        else:
            p_state = yf[0, :].copy()
            ac = 1.0
            sn2 = p_state.copy()
            pb = p_state.copy()
            pb2 = pb ** 2
            pminu = p_state.copy()
            actmin = np.full(nrf, np.inf)
            actminsub = actmin.copy()
            subwc = nv
            actbuf = np.full((nu, nrf), np.inf)
            ibuf = 0
            lminflag = np.zeros(nrf, dtype=bool)

    for t in range(nr):
        yft = yf[t, :]
        acb = (1.0 + (np.sum(p_state) / np.sum(yft) - 1.0) ** 2) ** (-1)
        ac = aca * ac + (1.0 - aca) * max(acb, acmax)
        ah = amax * ac * (1.0 + (p_state / sn2 - 1.0) ** 2) ** (-1)
        snr = np.sum(p_state) / np.sum(sn2)
        ah = np.maximum(ah, min(aminh, snr ** snrexp))

        p_state = ah * p_state + (1.0 - ah) * yft
        b_coeff = np.minimum(ah ** 2, bmax)
        pb = b_coeff * pb + (1.0 - b_coeff) * p_state
        pb2 = b_coeff * pb2 + (1.0 - b_coeff) * p_state ** 2

        qeqi = np.clip((pb2 - pb ** 2) / (2.0 * sn2 ** 2), qeqimin / (t + nrcum + 1), qeqimax)
        qiav = np.sum(qeqi) / nrf
        bc = 1.0 + av * np.sqrt(qiav)
        bmind = 1.0 + 2.0 * (nd - 1) * (1.0 - md) / (1.0 / qeqi - 2.0 * md)
        bminv = 1.0 + 2.0 * (nv - 1) * (1.0 - mv) / (1.0 / qeqi - 2.0 * mv)
        kmod = bc * p_state * bmind < actmin
        if np.any(kmod):
            actmin[kmod] = bc * p_state[kmod] * bmind[kmod]
            actminsub[kmod] = bc * p_state[kmod] * bminv[kmod]

        if subwc > 1 and subwc < nv:
            lminflag = lminflag | kmod
            pminu = np.minimum(actminsub, pminu)
            sn2 = pminu.copy()
        else:
            if subwc >= nv:
                ibuf = 1 + (ibuf % nu)  # 1-based circular index (MATLAB convention)
                actbuf[ibuf - 1, :] = actmin  # convert to 0-based for array access
                pminu = np.min(actbuf, axis=0)
                # find first qith > qiav
                ii = np.where(qiav < qith)[0]
                if len(ii) > 0:
                    nsm = nsms[ii[0]]
                else:
                    nsm = nsms[-1]
                lmin = lminflag & ~kmod & (actminsub < nsm * pminu) & (actminsub > pminu)
                if np.any(lmin):
                    pminu[lmin] = actminsub[lmin]
                    actbuf[:, lmin] = pminu[lmin]
                lminflag[:] = False
                actmin[:] = np.inf
                subwc = 0

        subwc += 1
        x[t, :] = sn2
        qisq = np.sqrt(qeqi)
        xs[t, :] = sn2 * np.sqrt(0.266 * (nd + 100 * qisq) * qisq / (1.0 + 0.005 * nd + 6.0 / nd) / (0.5 / qeqi + nd - 1))

    zo = {
        'nrcum': nrcum + nr,
        'p': p_state,
        'ac': ac,
        'sn2': sn2,
        'pb': pb,
        'pb2': pb2,
        'pminu': pminu,
        'actmin': actmin,
        'actminsub': actminsub,
        'subwc': subwc,
        'actbuf': actbuf,
        'ibuf': ibuf,
        'lminflag': lminflag,
        'tinc': tinc,
        'qq': qq,
    }
    return x, zo, xs
