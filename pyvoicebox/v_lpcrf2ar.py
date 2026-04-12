"""V_LPCRF2AR - Convert reflection coefficients to autoregressive coefficients."""

from __future__ import annotations
import numpy as np


def v_lpcrf2ar(rf) -> np.ndarray:
    """Convert reflection coefficients to autoregressive coefficients.

    Parameters
    ----------
    rf : array_like, shape (nf, p+1)
        Reflection coefficients.

    Returns
    -------
    ar : ndarray, shape (nf, p+1)
        Autoregressive coefficients with ar[:, 0] = 1.
    """
    rf = np.atleast_2d(np.asarray(rf, dtype=float))
    nf, p1 = rf.shape
    p = p1 - 1

    if p == 0:
        return np.ones((nf, 1))

    # arf: forward filter, arr: reverse filter
    arf = np.zeros((nf, p1))
    arf[:, 0] = 1.0
    arr = np.zeros((nf, p1))
    arr[:, p] = rf[:, p]  # arr(:,p1) in MATLAB (1-based)
    cr = np.zeros((nf, p))

    for k in range(1, p):
        # MATLAB: rk=rf(:,(p1-k)*ones(1,k)); -> rf column index p1-k (1-based) = p-k (0-based)
        rk = rf[:, p - k:p - k + 1]  # shape (nf, 1), broadcast later

        # MATLAB: cr(:,1:k)=arr(:,p2-k:p1);  p2=p1+1, so p2-k = p1+1-k (1-based) = p-k (0-based)
        # arr columns p-k to p (0-based), that's k+1 columns... wait
        # MATLAB p2-k:p1 in 1-based: (p1+1-k):p1, that's k elements
        # 0-based: (p1-k):(p1-1) inclusive, i.e. (p-k+1-1)=p-k to p-1... wait
        # p1+1-k to p1 in 1-based = p1-k to p1-1 in 0-based = indices p-k to p (0-based, last p1 col is index p)
        # MATLAB arr(:,p2-k:p1) where p2=p1+1: indices p1+1-k to p1, 1-based
        # 0-based: p1-k to p1-1, which is p-k+1-1? No.
        # p1 = p+1. MATLAB 1-based indices: p1+1-k = p+2-k to p1 = p+1.
        # That's k elements: p+2-k, p+3-k, ..., p+1
        # 0-based: p+1-k, p+2-k, ..., p. That's k elements.
        cr[:, :k] = arr[:, p + 1 - k:p + 1]

        # MATLAB: arr(:,p1-k:p)=arr(:,p1-k:p)+rk.*arf(:,1:k);
        # p1-k in 1-based = p-k in 0-based. p in 1-based = p-1 in 0-based
        # So arr columns p-k to p-1 (0-based), k elements
        arr[:, p - k:p] = arr[:, p - k:p] + rk * arf[:, :k]

        # MATLAB: arf(:,2:k+1)=arf(:,2:k+1)+rk.*cr(:,1:k);
        # 2 to k+1 in 1-based = 1 to k in 0-based, k elements
        arf[:, 1:k + 1] = arf[:, 1:k + 1] + rk * cr[:, :k]

    r1 = rf[:, 0:1]  # shape (nf, 1)
    ar = arf + r1 * arr

    return ar
