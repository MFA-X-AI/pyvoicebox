"""V_MOMFILT - Calculate moments of a signal using a sliding window."""

import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import hamming


def v_momfilt(x, r, w=None, m=None):
    """Calculate moments of a signal using a sliding window.

    Parameters
    ----------
    x : array_like
        Input signal.
    r : array_like
        List of moments to calculate (+ve relative to mean, -ve relative to zero).
    w : int or array_like, optional
        Window or window length. Default: Hamming of length(x).
    m : int, optional
        Center sample of window (1-based). Default: ceil((1+len(w))/2).

    Returns
    -------
    y : ndarray, shape (len(x), len(r))
        Moments.
    mm : int
        Actual value of m used.
    """
    x = np.asarray(x, dtype=float).ravel()
    r = np.round(np.asarray(r, dtype=float)).astype(int).ravel()

    if w is None:
        w = hamming(len(x), sym=True)
    elif np.isscalar(w):
        w = hamming(int(w), sym=True)
    else:
        w = np.asarray(w, dtype=float).ravel()

    lw = len(w)
    if m is None:
        m = (1 + lw) / 2.0
    m = max(int(round(m)), 1)
    mm = m

    lx = len(x)
    lxx = lx + m - 1
    xx = np.zeros(lxx)
    xx[:lx] = x

    # Compute y0 = filter(w, 1, ones)
    cw = np.cumsum(w)
    sw = cw[-1]
    y0 = np.full(lxx, sw)
    lxw = min(lxx, lw)
    y0[:lxw] = cw[:lxw]
    if m > 1:
        y0[lx:lx + m - 1] -= cw[:m - 1]
    yd = y0[m - 1:]
    yd[np.abs(yd) < np.finfo(float).eps] = 1.0

    nr = len(r)
    y = np.zeros((lx, nr))

    mr = max(np.abs(r)) if len(r) > 0 else 0
    mk = np.zeros(mr, dtype=int)
    neg_r = r[r < 0]
    if len(neg_r) > 0:
        mk[-neg_r - 1] = 1
    maxr = max(r) if len(r) > 0 else 0
    if maxr > 1:
        mk[:maxr] = 1

    ml = np.where(mk > 0)[0]  # 0-based indices of needed moments
    lml = len(ml)

    if lml > 0:
        moments_needed = ml + 1  # actual moment orders (1-based)
        xm = np.zeros((lxx, lml))
        for i, mom in enumerate(moments_needed):
            xm[:, i] = lfilter(w, [1.0], xx ** mom)
        xm = xm[m - 1:, :] / yd[:, np.newaxis]

        # Build mapping from moment order to column index
        mlx = np.zeros(mr, dtype=int)
        for i, mi in enumerate(ml):
            mlx[mi] = i

    # Fill in zero-centered moments (negative r)
    fr = np.where(r < 0)[0]
    if len(fr) > 0:
        for fi in fr:
            y[:, fi] = xm[:, mlx[-r[fi] - 1]]

    # 0th moment
    fr = np.where(r == 0)[0]
    if len(fr) > 0:
        y[:, fr] = 1.0

    # 1st moment about mean
    fr = np.where(r == 1)[0]
    if len(fr) > 0:
        y[:, fr] = 0.0

    # 2nd moment about mean (variance)
    fr = np.where(r == 2)[0]
    if len(fr) > 0:
        yfr = xm[:, mlx[1]] - xm[:, mlx[0]] ** 2
        for fi in fr:
            y[:, fi] = yfr

    # Higher moments about the mean
    if maxr > 2 and lml > 0:
        bc = np.array([1, -2, 1], dtype=float)
        mon = np.array([1, -1], dtype=float)
        am = np.zeros((lx, maxr))
        am[:, 0] = xm[:, mlx[0]]
        for mi in range(1, maxr):
            am[:, mi] = xm[:, mlx[0]] ** (mi + 1)

        for i in range(3, maxr + 1):
            bc = np.convolve(bc, mon)
            fr = np.where(r == i)[0]
            if len(fr) > 0:
                yfr = xm[:, mlx[i - 1]]
                for j in range(1, i - 1):
                    yfr = yfr + xm[:, mlx[i - j - 2]] * am[:, j - 1] * bc[j]
                yfr = yfr + am[:, i - 1] * (bc[i - 1] + bc[i])
                for fi in fr:
                    y[:, fi] = yfr

    return y, mm
