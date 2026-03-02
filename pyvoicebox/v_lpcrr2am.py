"""V_LPCRR2AM - Convert autocorrelation coefficients to AR coefficient matrix."""

import numpy as np


def v_lpcrr2am(rr):
    """Convert autocorrelation coefficients to AR coefficient matrix.

    Parameters
    ----------
    rr : array_like, shape (nf, p+1)
        Autocorrelation coefficients.

    Returns
    -------
    am : ndarray, shape (p+1, p+1, nf)
        AR coefficient matrix.
    em : ndarray, shape (nf, p+1)
        Residual energy for each order.
    """
    rr = np.atleast_2d(np.asarray(rr, dtype=float))
    nf, p1 = rr.shape
    p = p1 - 1

    am = np.zeros((nf, p1, p1))
    em = np.zeros((nf, p1))
    am[:, p, p] = 1.0
    em[:, p] = rr[:, 0]

    ar = np.ones((nf, p1))
    ar[:, 1] = -rr[:, 1] / rr[:, 0]
    e = rr[:, 0] * (ar[:, 1] ** 2 - 1)

    for n in range(2, p + 1):
        q = p1 - n
        em[:, q] = -e
        am[:, q:p1, q] = ar[:, :n]
        k = (rr[:, n] + np.sum(rr[:, n-1:0:-1] * ar[:, 1:n], axis=1)) / e
        ar[:, 1:n] = ar[:, 1:n] + k[:, np.newaxis] * ar[:, n-1:0:-1]
        ar[:, n] = k
        e = e * (1 - k ** 2)

    em[:, 0] = -e
    am[:, :, 0] = ar

    # Permute to (p+1, p+1, nf)
    am = np.transpose(am, (2, 1, 0))

    return am, em
