"""V_TEAGER - Calculate Teager energy waveform."""

import numpy as np


def v_teager(x, d=None, m=''):
    """Calculate Teager energy waveform.

    y(n) = abs(x(n))^2 - x(n+1)*conj(x(n-1))

    Parameters
    ----------
    x : array_like
        Input signal.
    d : int or None, optional
        Dimension to apply filter along (0-based).
        Default is first non-singleton dimension.
    m : str, optional
        'x' to suppress extrapolation of end points. Output will be
        two samples shorter than input.

    Returns
    -------
    y : ndarray
        Teager energy waveform.
    """
    x = np.asarray(x, dtype=float)
    e = x.shape
    p = x.size

    if d is None:
        d_found = None
        for i, si in enumerate(e):
            if si > 1:
                d_found = i
                break
        d = d_found if d_found is not None else 0

    k = e[d]  # size of active dimension
    q = p // k  # size of remainder

    # Move working dimension to front and reshape to 2D
    z = np.moveaxis(x, d, 0)
    r = z.shape
    z = z.reshape(k, q)

    if 'x' in m:
        y = z[1:k - 1, :] * np.conj(z[1:k - 1, :]) - z[2:k, :] * np.conj(z[0:k - 2, :])
        k_out = k - 2
    elif k >= 4:
        y = np.zeros((k, q))
        y[1:k - 1, :] = z[1:k - 1, :] * np.conj(z[1:k - 1, :]) - z[2:k, :] * np.conj(z[0:k - 2, :])
        y[0, :] = 2 * y[1, :] - y[2, :]
        y[k - 1, :] = 2 * y[k - 2, :] - y[k - 3, :]
        k_out = k
    elif k == 3:
        val = z[1, :] * np.conj(z[1, :]) - z[2, :] * np.conj(z[0, :])
        y = np.tile(val, (3, 1))
        k_out = k
    else:
        y = np.zeros((k, q))
        k_out = k

    # Reshape back
    r_out = list(r)
    r_out[0] = k_out
    y = y.reshape(r_out)
    y = np.moveaxis(y, 0, d)

    return np.real(y)
