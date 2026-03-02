"""V_LPCAA2RF - Convert vocal tract areas to reflection coefficients."""

import numpy as np


def v_lpcaa2rf(aa):
    """Convert vocal tract areas to reflection coefficients.

    Parameters
    ----------
    aa : array_like, shape (nf, p+2)
        Vocal tract areas.

    Returns
    -------
    rf : ndarray, shape (nf, p+1)
        Reflection coefficients.
    """
    aa = np.atleast_2d(np.asarray(aa, dtype=float))
    nf, p2 = aa.shape
    rf = (aa[:, 1:p2] - aa[:, :p2 - 1]) / (aa[:, 1:p2] + aa[:, :p2 - 1])
    return rf
