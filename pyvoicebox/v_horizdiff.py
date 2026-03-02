"""V_HORIZDIFF - Estimate horizontal difference between two functions."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


def v_horizdiff(y, v, x=None, u=None, q=''):
    """Estimate horizontal difference between two functions of x.

    Approximately: y(x) = v(x+z).

    Parameters
    ----------
    y : array_like, shape (n, m)
        Each column is a function of x.
    v : array_like, shape (k,)
        Reference function.
    x : array_like, shape (n,), optional
        x values for y. Default (1:n).
    u : array_like, shape (k,), optional
        x values for v. Default same as x.
    q : str, optional
        Interpolation mode: 'l' linear, 'p' pchip, 's' spline.

    Returns
    -------
    z : ndarray, shape (n, m)
        Horizontal difference.
    zm : ndarray, shape (m,)
        MMSE horizontal difference.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if y.shape[0] == 1 and y.shape[1] > 1:
        y = y.T
    n, m_cols = y.shape
    v = np.asarray(v, dtype=float).ravel()

    if x is None:
        x = np.arange(1, n + 1, dtype=float)
    else:
        x = np.asarray(x, dtype=float).ravel()

    if u is None:
        u = x.copy()
    else:
        u = np.asarray(u, dtype=float).ravel()

    # Choose interpolation method
    if n >= 4 and 's' in q:
        kind = 'cubic'
    elif n >= 2 and ('s' in q or 'p' in q):
        kind = 'cubic'
    else:
        kind = 'linear'

    # Interpolate inverse function u(v)
    f_inv = interp1d(v, u, kind=kind, fill_value='extrapolate')
    z = f_inv(y) - x[:, np.newaxis]

    # MMSE horizontal difference
    zm = np.zeros(m_cols)
    f_fwd = interp1d(u, v, kind=kind, fill_value='extrapolate')
    for i in range(m_cols):
        def objective(zm_val):
            return np.sum((y[:, i] - f_fwd(x + zm_val)) ** 2)
        result = minimize_scalar(objective, bounds=(u[0] - x[-1], u[-1] - x[0]), method='bounded')
        zm[i] = result.x

    return z, zm
