"""V_POW2CEP - Convert power domain means and variances to the cepstral domain."""

import numpy as np
from .v_rdct import v_rdct


def v_pow2cep(m, c, mode='c'):
    """Convert power domain means and variances to the cepstral domain.

    Parameters
    ----------
    m : array_like
        Vector giving means in the power domain.
    c : array_like
        Covariance matrix in the power domain (or diag(c) if diagonal).
    mode : str, optional
        'c' : pow=exp(irdct(cep)) [default]
        'i' : pow=exp(cep) [no transformation]

    Returns
    -------
    u : ndarray
        Row vector giving the cepstral means with u[0] the 0th cepstral coefficient.
    v : ndarray
        Cepstral covariance matrix.
    """
    m = np.asarray(m, dtype=float).ravel()
    c = np.asarray(c, dtype=float)
    if c.ndim == 1 or min(c.shape) == 1:
        c = np.diag(c.ravel())

    q = np.log(1.0 + c / np.outer(m, m))
    p = np.log(m) - 0.5 * np.diag(q)

    if 'i' in mode:
        u = p
        v = q
    else:
        # Default: DCT domain
        u = v_rdct(p)
        v = v_rdct(v_rdct(q).T)

    return u, v
