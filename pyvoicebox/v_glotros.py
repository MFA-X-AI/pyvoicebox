"""V_GLOTROS - Rosenberg glottal model."""

import numpy as np


def v_glotros(d, t=None, p=None):
    """Rosenberg glottal model.

    Parameters
    ----------
    d : int
        Derivative order (0, 1, or 2).
    t : array_like, optional
        Time in fractions of a cycle. Default: (0:99)/100.
    p : array_like, optional
        Parameters: p[0]=closure time, p[1]=+ve/-ve slope ratio.
        Default: [0.6, 0.5].

    Returns
    -------
    u : ndarray
        Output waveform (derivative of flow waveform if d>0).
    """
    if t is None:
        tt = np.arange(100) / 100.0
    else:
        tt = np.mod(np.asarray(t, dtype=float), 1.0)

    u = np.zeros_like(tt)
    de = np.array([0.6, 0.5])
    if p is None:
        p = de.copy()
    else:
        p = np.asarray(p, dtype=float).ravel()
        if len(p) < 2:
            p = np.concatenate([p, de[len(p):2]])

    pp = p[0] / (1.0 + p[1])
    ta = tt < pp
    tb = (tt < p[0]) & ~ta
    wa = np.pi / pp
    wb = 0.5 * np.pi / (p[0] - pp)
    fb = wb * pp

    if d == 0:
        u[ta] = 0.5 * (1.0 - np.cos(wa * tt[ta]))
        u[tb] = np.cos(wb * tt[tb] - fb)
    elif d == 1:
        u[ta] = 0.5 * wa * np.sin(wa * tt[ta])
        u[tb] = -wb * np.sin(wb * tt[tb] - fb)
    elif d == 2:
        u[ta] = 0.5 * wa**2 * np.cos(wa * tt[ta])
        u[tb] = -wb**2 * np.cos(wb * tt[tb] - fb)
    else:
        raise ValueError('Derivative must be 0, 1, or 2')

    return u
