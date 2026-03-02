"""V_LPCAA2DL - Convert area coefficients to DCT of log area."""

import numpy as np
from pyvoicebox.v_rdct import v_rdct


def v_lpcaa2dl(aa):
    """Convert area coefficients to DCT of log area.

    Parameters
    ----------
    aa : array_like, shape (nf, p+2)
        Area coefficients.

    Returns
    -------
    dl : ndarray, shape (nf, p)
        DCT of log area.
    """
    aa = np.atleast_2d(np.asarray(aa, dtype=float))
    nf, p2 = aa.shape
    # log(aa(:,2:p2-1) ./ aa(:,p2*ones(1,p2-2)))
    inner = aa[:, 1:p2 - 1] / aa[:, -1:]
    dl = v_rdct(np.log(inner).T).T
    return dl
