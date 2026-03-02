"""V_LPCDL2AA - Convert DCT of log area to area coefficients."""

import numpy as np
from pyvoicebox.v_irdct import v_irdct


def v_lpcdl2aa(dl):
    """Convert DCT of log area to area coefficients.

    Parameters
    ----------
    dl : array_like, shape (nf, p)
        DCT of log area.

    Returns
    -------
    aa : ndarray, shape (nf, p+2)
        Area coefficients with aa[:, 0] = 0 and aa[:, -1] = 1.
    """
    dl = np.atleast_2d(np.asarray(dl, dtype=float))
    nf, p = dl.shape
    aa = np.column_stack([
        np.zeros((nf, 1)),
        np.exp(v_irdct(dl.T).T),
        np.ones((nf, 1))
    ])
    return aa
