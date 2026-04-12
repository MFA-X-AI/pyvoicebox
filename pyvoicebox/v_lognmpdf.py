"""V_LOGNMPDF - Calculate PDF of a multivariate lognormal distribution."""

from __future__ import annotations
import numpy as np
from scipy.stats import multivariate_normal


def v_lognmpdf(x, m=None, v=None) -> np.ndarray:
    """Calculate PDF of a multivariate lognormal distribution.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate, shape (n, d).
    m : array_like, optional
        Mean vector (d,). Default: ones.
    v : array_like, optional
        Covariance matrix (d, d) or diagonal vector (d,). Default: identity.

    Returns
    -------
    p : ndarray
        PDF values, shape (n,).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape

    if m is None:
        m = np.ones(d)
    else:
        m = np.asarray(m, dtype=float).ravel()

    if v is None:
        v = np.eye(d)
    else:
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            v = np.diag(v)

    # Convert from natural domain mean/covariance to log domain
    # Using the standard lognormal parameterization:
    # If X ~ LogN(mu, sigma), then E[X] = exp(mu + sigma^2/2)
    # For multivariate case: mu_log and sigma_log are the parameters
    # of the underlying normal distribution.
    #
    # Given natural mean m and covariance v:
    # sigma_log_ij = log(1 + v_ij / (m_i * m_j))
    # mu_log_i = log(m_i) - 0.5 * sigma_log_ii

    sigma_log = np.log(1 + v / np.outer(m, m))
    mu_log = np.log(m) - 0.5 * np.diag(sigma_log)

    p = np.zeros(n)
    c = np.prod(x, axis=1)
    q = c > 0
    if np.any(q):
        log_x = np.log(x[q, :])
        rv = multivariate_normal(mean=mu_log, cov=sigma_log)
        p[q] = rv.pdf(log_x) / c[q]

    return p
