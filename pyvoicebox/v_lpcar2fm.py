"""V_LPCAR2FM - Convert autoregressive coefficients to formant freq+amp+bw."""

import numpy as np
from pyvoicebox.v_lpcar2zz import v_lpcar2zz


def v_lpcar2fm(ar, t=None):
    """Convert autoregressive coefficients to formant frequencies, amplitudes and bandwidths.

    Parameters
    ----------
    ar : array_like, shape (nf, p+1)
        Autoregressive coefficients.
    t : float, optional
        Threshold. If t>0, poles with bandwidth > t*frequency are ignored.
        If t<=0, poles with bandwidth > -t are ignored.

    Returns
    -------
    n : ndarray, shape (nf,)
        Number of formants found per frame.
    f : ndarray, shape (nf, m)
        Formant frequencies in normalized Hz.
    a : ndarray, shape (nf, m)
        Formant amplitudes.
    b : ndarray, shape (nf, m)
        Formant bandwidths in normalized Hz.
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape
    p = p1 - 1

    zz = v_lpcar2zz(ar)
    ig = np.imag(zz) <= 0
    n = p - np.sum(ig, axis=1)
    mn = int(np.max(n))

    # Remove redundant columns
    if mn < p:
        # Sort so that ig=True comes first (non-formant poles)
        ix = np.argsort(ig.astype(int), axis=1)
        zz_new = np.zeros((nf, mn), dtype=complex)
        ig_new = np.zeros((nf, mn), dtype=bool)
        for i in range(nf):
            zz_new[i, :] = zz[i, ix[i, :mn]]
            ig_new[i, :] = ig[i, ix[i, :mn]]
        zz = zz_new
        ig = ig_new

    zz_safe = zz.copy()
    zz_safe[ig] = 1.0  # prevent infinities
    f = np.angle(zz_safe) * 0.5 / np.pi
    b = -np.log(np.abs(zz_safe)) / np.pi

    if t is not None:
        if t > 0:
            ig = ig | (b > t * f)
        else:
            ig = ig | (b + t > 0)

    f[ig] = 0.0
    b[ig] = 0.0
    n = f.shape[1] - np.sum(ig, axis=1) if f.size > 0 else np.zeros(nf, dtype=int)
    m = int(np.max(n)) if np.any(n > 0) else 0

    if m == 0:
        return n, np.zeros((nf, 0)), np.zeros((nf, 0)), np.zeros((nf, 0))

    # Sort: non-ignored values first, by frequency
    sort_key = ig.astype(float) + f
    ix = np.argsort(sort_key, axis=1)

    zz_out = np.zeros((nf, m), dtype=complex)
    f_out = np.zeros((nf, m))
    b_out = np.zeros((nf, m))
    ig_out = np.zeros((nf, m), dtype=bool)
    for i in range(nf):
        zz_out[i, :] = zz[i, ix[i, :m]]
        f_out[i, :] = f[i, ix[i, :m]]
        b_out[i, :] = b[i, ix[i, :m]]
        ig_out[i, :] = ig[i, ix[i, :m]]

    # Calculate gain
    pw = -2j * np.pi * np.arange(p1)
    a_out = np.zeros((nf, m))
    for i in range(nf):
        for j in range(m):
            val = np.sum(ar[i, :] * np.exp(pw * f_out[i, j]))
            a_out[i, j] = np.abs(val) ** (-1)

    return n, f_out, a_out, b_out
