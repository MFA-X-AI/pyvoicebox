"""V_RESAMPLE - Resample and remove end transients."""

import numpy as np
from scipy.signal import resample_poly


def v_resample(x, p, q, n=10, b=5):
    """Resample signal and remove end transients.

    Parameters
    ----------
    x : array_like
        Input signal.
    p : int
        Upsampling factor.
    q : int
        Downsampling factor.
    n : int, optional
        Filter length. Default 10.
    b : float, optional
        Kaiser window beta. Default 5.

    Returns
    -------
    y : ndarray
        Resampled output signal.
    """
    x = np.asarray(x, dtype=float)
    if p == q:
        return x.copy()

    y = resample_poly(x, p, q)
    m = int(np.ceil(n * max(p / q, 1)))
    if x.ndim == 1:
        y = y[m:-m] if len(y) > 2 * m else y
    else:
        y = y[m:-m, :] if y.shape[0] > 2 * m else y
    return y
