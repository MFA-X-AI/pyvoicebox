"""V_PEAK2DQUAD - Find quadratically-interpolated peak in a 2D array."""

from pyvoicebox.v_quadpeak import v_quadpeak


def v_peak2dquad(z):
    """Find quadratically-interpolated peak in a 2D array.

    This is a wrapper around v_quadpeak.

    Parameters
    ----------
    z : array_like, shape (m, n)
        Input 2D array.

    Returns
    -------
    v : float
        Peak value.
    xy : ndarray
        Position of peak.
    t : int
        -1 for maximum, 0 for saddle point, +1 for minimum.
    m : ndarray
        Fitted quadratic matrix.
    """
    return v_quadpeak(z)
