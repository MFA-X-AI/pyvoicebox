"""V_LPCRF2AA - Convert reflection coefficients to area function."""

from __future__ import annotations
import numpy as np


def v_lpcrf2aa(rf) -> np.ndarray:
    """Convert reflection coefficients to area function.

    Parameters
    ----------
    rf : array_like, shape (nf, p+1)
        Reflection coefficients.

    Returns
    -------
    aa : ndarray, shape (nf, p+2)
        Area function, normalized so aa[:, -1] = 1.
    """
    rf = np.atleast_2d(np.asarray(rf, dtype=float))
    nf, p1 = rf.shape
    # MATLAB: aa = fliplr(cumprod([ones(1,size(rf,1)); fliplr((1-rf)./(1+rf)).']).')
    ratios = ((1 - rf) / (1 + rf))[:, ::-1]  # flip left-right
    # Prepend column of ones (for glottis boundary)
    ratios_with_ones = np.column_stack([np.ones((nf, 1)), ratios])
    # Cumulative product along columns
    aa = np.cumprod(ratios_with_ones, axis=1)
    # Flip left-right
    aa = aa[:, ::-1]
    return aa
