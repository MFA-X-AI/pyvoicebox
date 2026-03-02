"""V_LPCRA2AR - Convert inverse filter autocorrelation coefficients to AR filter."""

import numpy as np
from scipy.linalg import hankel, toeplitz


def v_lpcra2ar(ra, tol=1e-8):
    """Convert inverse filter autocorrelation coefficients to AR filter.

    Uses a Newton-Raphson iteration (Wilson's algorithm).

    Parameters
    ----------
    ra : array_like, shape (n, p+1)
        Each row is the second half of the autocorrelation of the
        coefficients of a stable AR filter of order p.
    tol : float, optional
        Tolerance relative to ra[0]. Default is 1e-8.

    Returns
    -------
    ar : ndarray, shape (n, p+1)
        AR filter coefficients.
    """
    ra = np.atleast_2d(np.asarray(ra, dtype=float))
    imax = 20
    nf, pp = ra.shape

    ar = np.zeros((nf, pp))
    for n in range(nf):
        xa = ra[n, :].copy()
        ax = np.zeros(pp)
        ax[0] = np.sqrt(xa[0] + 2 * np.sum(xa[1:]))

        i = imax
        while i > 0:
            t1 = hankel(ax)
            t2 = toeplitz(ax, np.concatenate([[ax[0]], np.zeros(pp - 1)]))
            ct = ax @ t1
            # MATLAB: ax = (xa+ct)/(t1+t2) means ax * (t1+t2) = (xa+ct)
            # Equivalent to solving (t1+t2)^T * ax^T = (xa+ct)^T
            ax = np.linalg.solve((t1 + t2).T, xa + ct)
            err = np.max(np.abs(ct - xa))
            if err <= tol * xa[0]:
                i = min(i - 1, 1)  # do one more iteration
            else:
                i -= 1

        ar[n, :] = ax

    return ar
