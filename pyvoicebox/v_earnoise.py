"""V_EARNOISE - Add noise to simulate hearing threshold."""

import numpy as np


def v_earnoise(s, fs, m='', spl=62.35):
    """Add noise to simulate the hearing threshold of a listener.

    Simplified version that adds white noise scaled to simulate internal ear noise.

    Parameters
    ----------
    s : array_like, shape (n,) or (n, c)
        Speech signal.
    fs : float
        Sample frequency in Hz.
    m : str, optional
        Mode string: 'n' input normalized, 'u' input already scaled.
    spl : float, optional
        Target active speech level in dB SPL. Default 62.35.

    Returns
    -------
    y : ndarray
        Speech with simulated ear noise.
    x : ndarray
        Filtered speech signal.
    v : ndarray
        Added noise.
    """
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s[:, np.newaxis]

    ns, nc = s.shape

    if 'u' in m:
        dboff = 0.0
    elif 'n' in m:
        dboff = spl
    else:
        dboff = spl  # simplified: assume 0 dB input

    x = 10 ** (0.05 * dboff) * s
    v = np.sqrt(0.5 * fs) * np.random.randn(*s.shape)
    y = x + v

    if nc == 1:
        return y.ravel(), x.ravel(), v.ravel()
    return y, x, v
