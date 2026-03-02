"""V_GLOTLF - Liljencrants-Fant glottal model."""

import numpy as np


def v_glotlf(d=0, t=None, p=None):
    """Liljencrants-Fant glottal model.

    Parameters
    ----------
    d : int, optional
        Derivative order (0, 1, or 2). Default: 0.
    t : array_like, optional
        Time in fractions of a cycle. Default: (0:99)/100.
    p : array_like, optional
        Parameters [te, E0/Ee, 1-tp/te]. Default: [0.6, 0.1, 0.2].

    Returns
    -------
    u : ndarray
        Output waveform.
    q : dict
        Structure with glottal model parameters.
    """
    if t is None:
        tt = np.arange(100) / 100.0
    else:
        t = np.asarray(t, dtype=float)
        tt = t - np.floor(t)

    u = np.zeros_like(tt)
    de = np.array([0.6, 0.1, 0.2])
    if p is None:
        p = de.copy()
    else:
        p = np.asarray(p, dtype=float).ravel()
        if len(p) < 3:
            p = np.concatenate([p, de[len(p):3]])

    # Calculate parameters
    te = p[0]
    mtc = te - 1.0
    e0 = 1.0
    wa = np.pi / (te * (1.0 - p[2]))
    a = -np.log(-p[1] * np.sin(wa * te)) / te
    inta = e0 * ((wa / np.tan(wa * te) - a) / p[1] + wa) / (a**2 + wa**2)

    rb0 = p[1] * inta
    rb = rb0

    # Newton iteration for closure time constant
    thresh = 1e-9
    err = 1.0
    for _ in range(15):
        kk = 1.0 - np.exp(mtc / rb)
        err = rb + mtc * (1.0 / kk - 1.0) - rb0
        derr = 1.0 - (1.0 - kk) * (mtc / rb / kk)**2
        rb = rb - err / derr
        if abs(err) < thresh:
            break

    if abs(err) > thresh:
        raise ValueError('Requested glottal waveform parameters are not feasible')

    e1 = 1.0 / (p[1] * (1.0 - np.exp(mtc / rb)))
    ta_mask = tt < te
    tb_mask = ~ta_mask

    if d == 0:
        u[ta_mask] = e0 * (np.exp(a * tt[ta_mask]) * (a * np.sin(wa * tt[ta_mask]) - wa * np.cos(wa * tt[ta_mask])) + wa) / (a**2 + wa**2)
        u[tb_mask] = e1 * (np.exp(mtc / rb) * (tt[tb_mask] - 1.0 - rb) + np.exp((te - tt[tb_mask]) / rb) * rb)
    elif d == 1:
        u[ta_mask] = e0 * np.exp(a * tt[ta_mask]) * np.sin(wa * tt[ta_mask])
        u[tb_mask] = e1 * (np.exp(mtc / rb) - np.exp((te - tt[tb_mask]) / rb))
    elif d == 2:
        u[ta_mask] = e0 * np.exp(a * tt[ta_mask]) * (a * np.sin(wa * tt[ta_mask]) + wa * np.cos(wa * tt[ta_mask]))
        u[tb_mask] = e1 * np.exp((te - tt[tb_mask]) / rb) / rb
    else:
        raise ValueError('Derivative must be 0, 1, or 2')

    # Build parameter structure
    ti = (np.pi + np.arctan(-wa / a)) / wa
    tp = np.pi / wa
    q = {}
    q['Up'] = e0 * wa * (np.exp(a * tp) + 1.0) / (a**2 + wa**2)
    q['E0'] = 1.0
    q['Ei'] = e0 * np.exp(a * ti) * np.sin(wa * ti)
    q['Ee'] = 1.0 / p[1]
    q['alpha'] = a
    q['epsilon'] = 1.0 / rb
    q['omega'] = wa
    q['t0'] = 0.0
    q['ti'] = ti
    q['tp'] = tp
    q['te'] = te
    q['ta'] = rb / (p[1] * e1)
    q['tc'] = 1.0
    q['Utc'] = -err / p[1]

    return u, q
