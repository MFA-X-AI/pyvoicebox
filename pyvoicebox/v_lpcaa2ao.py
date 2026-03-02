"""V_LPCAA2AO - Convert area function to area ratios."""

import numpy as np


def v_lpcaa2ao(aa):
    """Convert area function to area ratios.

    Parameters
    ----------
    aa : array_like, shape (nf, p+2)
        Area function.

    Returns
    -------
    ao : ndarray, shape (nf, p+1)
        Area ratios.
    """
    aa = np.atleast_2d(np.asarray(aa, dtype=float))
    p1 = aa.shape[1]
    ao = aa[:, :p1 - 1] / aa[:, 1:p1]
    return ao
