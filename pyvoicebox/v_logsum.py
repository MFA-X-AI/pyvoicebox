"""V_LOGSUM - log(sum(k.*exp(x),d)) computed avoiding overflow/underflow."""

from __future__ import annotations
import numpy as np
from ._compat import first_nonsingleton


def v_logsum(x, d=None, k=None) -> np.ndarray:
    """Compute log(sum(k.*exp(x), d)) avoiding overflow/underflow.

    Parameters
    ----------
    x : array_like
        Data array.
    d : int, optional
        Axis to sum along. Default: first non-singleton dimension.
    k : array_like, optional
        Scaling array (same shape as x, or vector along axis d).

    Returns
    -------
    y : ndarray
        log(sum(k.*exp(x), d))
    """
    x = np.asarray(x, dtype=float)

    if d is None:
        d = first_nonsingleton(x)

    n = x.shape[d]
    if n <= 1:
        if k is None:
            return x.copy()
        else:
            return x + np.log(np.asarray(k, dtype=float))

    q = np.max(x, axis=d, keepdims=True)

    # Handle infinities
    a = np.isinf(q)

    if k is None:
        y = q + np.log(np.sum(np.exp(x - q), axis=d, keepdims=True))
    else:
        k = np.asarray(k, dtype=float)
        # Broadcast k to match x along axis d
        if k.ndim == 1 and k.shape[0] == n:
            shape = [1] * x.ndim
            shape[d] = n
            k = k.reshape(shape)
        y = q + np.log(np.sum(np.exp(x - q) * k, axis=d, keepdims=True))

    # Correct columns whose max is +-Inf
    y = np.where(a, q, y)

    # Remove the keepdims axis to match MATLAB behavior
    y = np.squeeze(y, axis=d)

    return y
