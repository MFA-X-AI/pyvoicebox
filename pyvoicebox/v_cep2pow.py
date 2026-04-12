"""V_CEP2POW - Convert cepstral means and variances to the power domain."""

from __future__ import annotations
import numpy as np
from .v_irdct import v_irdct


def v_cep2pow(u, v, mode='c') -> tuple[np.ndarray, np.ndarray]:
    """Convert cepstral means and variances to the power domain.

    Parameters
    ----------
    u : array_like
        Vector giving the cepstral means with u[0] the 0th cepstral coefficient.
    v : array_like
        Cepstral covariance matrix or vector containing the diagonal elements.
    mode : str, optional
        'c' : pow=exp(irdct(cep)) [default]
        'i' : pow=exp(cep) [no transformation]

    Returns
    -------
    m : ndarray
        Row vector giving means in the power domain.
    c : ndarray
        Covariance matrix in the power domain.
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float)
    if v.ndim == 1 or min(v.shape) == 1:
        v = np.diag(v.ravel())

    if 'i' in mode:
        p_vec = u.copy()
        q_mat = v.copy()
    else:
        # Default: DCT domain
        p_vec = v_irdct(u)
        q_mat = v_irdct(v_irdct(v).T)

    m = np.exp(p_vec + 0.5 * np.diag(q_mat))
    c = np.outer(m, m) * (np.exp(q_mat) - 1.0)
    return m, c
