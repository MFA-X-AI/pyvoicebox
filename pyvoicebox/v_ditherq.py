"""V_DITHERQ - Add dither and quantize."""

import numpy as np


def _matlab_round(x):
    """Round half away from zero (MATLAB-style rounding)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def v_ditherq(x, m='w', zi=None, rng=None):
    """Add dither and quantize.

    Parameters
    ----------
    x : array_like
        Input signal.
    m : str, optional
        Mode: 'w' white dither (default), 'h' high-pass dither,
        'l' low-pass dither, 'n' no dither.
    zi : float, optional
        Initial state for filtering. Default is random.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility. Default uses np.random.

    Returns
    -------
    y : ndarray
        Dithered and quantized signal.
    zf : float
        Output state.
    """
    x = np.asarray(x, dtype=float)
    was_row = (x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1))
    xflat = x.ravel()
    n = len(xflat)

    if rng is None:
        rng = np.random.default_rng()

    if zi is None:
        zi = rng.random()

    if 'n' in m:
        y = _matlab_round(xflat)
        zf = zi
    elif 'h' in m or 'l' in m:
        v = rng.random(n + 1)
        v[0] = zi
        zf = v[-1]
        if 'h' in m:
            y = _matlab_round(xflat + v[1:] - v[:-1])
        else:
            y = _matlab_round(xflat + v[1:] + v[:-1] - 1)
    else:
        # White dither (default)
        r = rng.random((n, 2))
        y = _matlab_round(xflat + r[:, 0] - r[:, 1])
        zf = rng.random()

    if was_row and x.ndim >= 2 and x.shape[0] == 1:
        y = y.reshape(1, -1)
    elif x.ndim == 0:
        y = y.reshape(())
    else:
        y = y.reshape(x.shape)

    return y, zf
