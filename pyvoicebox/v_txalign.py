"""V_TXALIGN - Find best alignment of two sets of time markers."""

from __future__ import annotations
import numpy as np


def v_txalign(x, y, maxt, nsd=None) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Find the best alignment of two sets of time markers.

    Parameters
    ----------
    x : array_like
        First set of non-decreasing time values.
    y : array_like
        Second set of non-decreasing time values.
    maxt : float
        Penalty threshold.
    nsd : float, optional
        If specified, threshold is nsd times std dev from mean.

    Returns
    -------
    kx : ndarray
        Alignment from x to y (kx[i]=j means x[i] matched to y[j], 0=unmatched).
    ky : ndarray
        Alignment from y to x.
    nxy : int
        Number of matched pairs.
    mxy : float
        Mean of y-x for matched pairs.
    sxy : float
        Std dev of y-x for matched pairs.
    """
    x = np.asarray(x, dtype=float).ravel().copy()
    y = np.asarray(y, dtype=float).ravel().copy()
    lx = len(x)
    ly = len(y)

    if lx == 0 or ly == 0:
        return np.zeros(lx, dtype=int), np.zeros(ly, dtype=int), 0, 0.0, 0.0

    if nsd is not None:
        kx, ky, nxy, mxy, sxy = v_txalign(x, y, maxt)
        nxy0 = nxy + 1
        while nxy < nxy0:
            nxy0 = nxy
            mxy0 = mxy
            kx, ky, nxy, mxy, sxy = v_txalign(x + mxy0, y, nsd * sxy)
        mxy = mxy + mxy0
        return kx, ky, nxy, mxy, sxy

    # Add sentinel values
    x_ext = np.append(x, 2 * abs(y[-1]) + 1)
    y_ext = np.append(y, 2 * abs(x[-1]) + 1)
    lxy = lx + ly + 1

    d = np.zeros((lxy, 4))
    c0 = maxt ** 2
    ix = 0  # 0-based index into x_ext
    iy = 0  # 0-based index into y_ext
    d[0, :] = [0, 0, -1, 1 if y_ext[0] < x_ext[0] else 0]
    vp = 0.0

    for id_ in range(1, lxy):
        if y_ext[iy] < x_ext[ix]:
            v = y_ext[iy]
            d[id_, 3] = 1
            iy += 1
        else:
            v = x_ext[ix]
            d[id_, 3] = 0
            ix += 1

        if d[id_, 3] == d[id_ - 1, 3]:
            d[id_, 2] = -1
        else:
            d[id_, 1] = d[id_ - 1, 0] - c0 + (v - vp) ** 2

        if d[id_ - 1, 2] == 0 and d[id_ - 1, 0] >= d[id_ - 1, 1]:
            d[id_, 0] = d[id_ - 1, 1]
            d[id_ - 1, 2] = 1
        else:
            d[id_, 0] = d[id_ - 1, 0]

        vp = v

    if d[lxy - 1, 2] == 0 and d[lxy - 1, 0] >= d[lxy - 1, 1]:
        d[lxy - 1, 2] = 1

    # Traceback
    ix = lx - 1
    iy = ly - 1
    nxy = 0
    mxy = 0.0
    sxy = 0.0
    kx = np.zeros(lx, dtype=int)
    ky = np.zeros(ly, dtype=int)

    while ix >= 0 and iy >= 0:
        id_ = ix + iy + 2  # 0-based in d array (was ix+iy+1 in 1-based)
        if d[id_, 2] > 0:
            ky[iy] = ix + 1  # 1-based output
            kx[ix] = iy + 1
            dt = y[iy] - x[ix]
            nxy += 1
            mxy += dt
            sxy += dt ** 2
            ix -= 1
            iy -= 1
        else:
            if d[id_, 3] == 1:
                iy -= 1
            else:
                ix -= 1

    if nxy > 0:
        mxy = mxy / nxy
        sxy = np.sqrt(sxy / nxy - mxy ** 2)

    return kx, ky, nxy, mxy, sxy
