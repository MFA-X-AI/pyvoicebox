"""V_PSYCHOFUNC - Calculate psychometric functions."""

import numpy as np
from scipy.special import erfc


def v_psychofunc(m_or_q=None, q=None, x=None, r=None):
    """Calculate psychometric functions: trial success probability vs SNR.

    Parameters
    ----------
    m_or_q : str or array_like
        Mode string or q parameters (if mode omitted).
    q : array_like, optional
        Model parameters [p_threshold, threshold, slope, miss_prob, guess_prob, type].
    x : array_like, optional
        SNR values or probability values (for inverse).
    r : array_like, optional
        Test results (0 or 1).

    Returns
    -------
    p : ndarray
        Probabilities.
    """
    # Parse arguments
    if isinstance(m_or_q, str):
        m = m_or_q
    elif m_or_q is not None:
        # Mode argument was omitted
        r = x
        x = q
        q = m_or_q
        m = ''
    else:
        m = ''

    qq = np.array([0.5, 0, 0.1, 0, 0, 1])  # defaults

    if q is not None:
        q = np.asarray(q, dtype=float).ravel()
        qq[:len(q)] = q[:min(len(q), 6)]

    pt = qq[0]    # probability at threshold
    xt = qq[1]    # threshold
    sl = qq[2]    # slope at threshold
    pm = qq[3]    # miss/lapse probability
    pg = qq[4]    # guess probability
    ft = int(qq[5])  # function type

    if x is None:
        x = np.linspace(xt - 20, xt + 20, 100)
    x = np.asarray(x, dtype=float)

    if 'i' in m:
        # Inverse function
        p = np.clip(x, pg + 1e-10, 1 - pm - 1e-10)
        u = (p - pg) / (1 - pm - pg)
        u = np.clip(u, 1e-15, 1 - 1e-15)

        if ft == 1:  # logistic
            ut = (pt - pg) / (1 - pm - pg)
            k = sl / (ut * (1 - ut) * (1 - pm - pg))
            return xt + (np.log(u / (1 - u)) - np.log(ut / (1 - ut))) / k
        elif ft == 2:  # cumulative Gaussian
            from scipy.special import erfinv
            ut = (pt - pg) / (1 - pm - pg)
            sig = (1 - pm - pg) / (sl * np.sqrt(2 * np.pi))
            return xt + sig * np.sqrt(2) * (erfinv(2 * u - 1) - erfinv(2 * ut - 1))
        else:
            raise NotImplementedError(f"Function type {ft} inverse not implemented")

    # Calculate normalized variable
    pr = 1 - pm - pg  # probability range

    if ft == 1:  # logistic
        ut = (pt - pg) / pr
        k = sl / (ut * (1 - ut) * pr)
        v = k * (x - xt)
        # logistic with offset
        vt = np.log(ut / (1 - ut))
        u = 1.0 / (1.0 + np.exp(-(v + vt)))
    elif ft == 2:  # cumulative Gaussian
        ut = (pt - pg) / pr
        from scipy.special import erfinv
        sig = pr / (sl * np.sqrt(2 * np.pi))
        z = (x - xt) / (sig * np.sqrt(2))
        zt = erfinv(2 * ut - 1)
        u = 0.5 * erfc(-(z + zt))
    elif ft == 3:  # Weibull
        ut = (pt - pg) / pr
        k = sl * np.exp(1) * np.log(1 / (1 - ut)) / pr
        u = 1 - np.exp(-np.exp(k * (x - xt) + np.log(np.log(1 / (1 - ut)))))
        u = np.clip(u, 0, 1)
    else:
        raise NotImplementedError(f"Psychometric function type {ft} not implemented")

    p = pg + pr * u

    if r is not None:
        r = np.asarray(r, dtype=float)
        if 'n' not in m:
            # Normalize likelihoods
            p = np.clip(p, 1e-15, 1 - 1e-15)
            likelihood = r * np.log(p) + (1 - r) * np.log(1 - p)
            return np.exp(likelihood - np.max(likelihood))

    if 'r' in m:
        return (np.random.rand(*p.shape) < p).astype(float)

    return p
