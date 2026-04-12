"""V_FINDPEAKS - Find peaks with optional quadratic interpolation."""

from __future__ import annotations
import numpy as np


def v_findpeaks(y, m='', w=None, x=None) -> tuple[np.ndarray, np.ndarray]:
    """Find peaks with optional quadratic interpolation.

    Parameters
    ----------
    y : array_like
        Input signal.
    m : str, optional
        Mode string:
          'f' - include first sample if downward initial slope
          'l' - include last sample if upward final slope
          'm' - return only the maximum peak
          'q' - quadratic interpolation
          'v' - find valleys instead of peaks
    w : float, optional
        Width tolerance. Peaks closer than w will have the lower one removed.
    x : array_like, optional
        X-axis values for y. Default: 0-based indices (1-based in MATLAB).

    Returns
    -------
    k : ndarray
        Positions of peaks (1-based if x not given, fractional if 'q').
    v : ndarray
        Peak amplitudes.
    """
    y = np.asarray(y, dtype=float).ravel()
    ny = len(y)

    if x is not None:
        x = np.asarray(x, dtype=float).ravel()

    if 'v' in m:
        y = -y.copy()
    else:
        y = y.copy()

    dx = y[1:] - y[:-1]
    r = np.where(dx > 0)[0]  # indices where signal rises
    f_idx = np.where(dx < 0)[0]  # indices where signal falls

    k = np.array([], dtype=float)
    v = np.array([], dtype=float)

    if len(r) > 0 and len(f_idx) > 0:
        # Convert to 1-based indices to match MATLAB logic throughout
        r1 = r + 1  # 1-based rise indices
        f1 = f_idx + 1  # 1-based fall indices

        # Compute rs: time since the last rise
        dr = r1.copy()
        dr[1:] = r1[1:] - r1[:-1]
        rc = np.ones(ny)
        rc[r1] = 1 - dr  # MATLAB: rc(r+1) where r is 1-based
        rc[0] = 0
        rs = np.cumsum(rc).astype(int)

        # Compute fs: time since the last fall
        df = f1.copy()
        df[1:] = f1[1:] - f1[:-1]
        fc = np.ones(ny)
        fc[f1] = 1 - df
        fc[0] = 0
        fs = np.cumsum(fc).astype(int)

        # Compute rq: time to the next rise
        # MATLAB: rp([1; r+1]) = [dr-1; ny-r(end)-1]
        # [1; r+1] in MATLAB (1-based) = [0; r1] in 0-based
        rp = -np.ones(ny)
        rp_indices = np.concatenate([[0], r1])
        rp_values = np.concatenate([dr - 1, [ny - r1[-1] - 1]])
        rp[rp_indices] = rp_values
        rq = np.cumsum(rp).astype(int)

        # Compute fq: time to the next fall
        fp = -np.ones(ny)
        fp_indices = np.concatenate([[0], f1])
        fp_values = np.concatenate([df - 1, [ny - f1[-1] - 1]])
        fp[fp_indices] = fp_values
        fq = np.cumsum(fp).astype(int)

        # Find peaks: (rs<fs) & (fq<rq) & centered in plateau
        k_idx = np.where(
            (rs < fs) & (fq < rq) &
            (np.floor((fq - rs) / 2.0) == 0)
        )[0]
        v = y[k_idx].copy()

        if 'q' in m:
            if x is not None:
                xm = x[k_idx - 1] - x[k_idx]
                xp = x[k_idx + 1] - x[k_idx]
                ym = y[k_idx - 1] - y[k_idx]
                yp = y[k_idx + 1] - y[k_idx]
                d_val = xm * xp * (xm - xp)
                b = 0.5 * (yp * xm ** 2 - ym * xp ** 2)
                a = xm * yp - xp * ym
                j = a > 0  # j=0 on a plateau
                v[j] = y[k_idx[j]] + b[j] ** 2 / (a[j] * d_val[j])
                k = np.empty_like(k_idx, dtype=float)
                k[j] = x[k_idx[j]] + b[j] / a[j]
                k[~j] = 0.5 * (x[k_idx[~j] + fq[k_idx[~j]]] +
                                x[k_idx[~j] - rs[k_idx[~j]]])
            else:
                # k_idx are 0-based; MATLAB uses 1-based indices
                k_1based = k_idx + 1  # convert to 1-based for computation
                b = 0.25 * (y[k_idx + 1] - y[k_idx - 1])
                a = y[k_idx] - 2 * b - y[k_idx - 1]
                j = a > 0
                k = k_1based.astype(float)
                v[j] = y[k_idx[j]] + b[j] ** 2 / a[j]
                k[j] = k_1based[j] + b[j] / a[j]
                k[~j] = k_1based[~j] + (fq[k_idx[~j]] - rs[k_idx[~j]]) / 2.0
        elif x is not None:
            k = x[k_idx].copy()
        else:
            k = (k_idx + 1).astype(float)  # 1-based

    # Add first and last samples if requested
    if ny > 1:
        if 'f' in m and y[0] > y[1]:
            v = np.concatenate([[y[0]], v])
            if x is not None:
                k = np.concatenate([[x[0]], k])
            else:
                k = np.concatenate([[1.0], k])

        if 'l' in m and y[ny - 2] < y[ny - 1]:
            v = np.concatenate([v, [y[ny - 1]]])
            if x is not None:
                k = np.concatenate([k, [x[ny - 1]]])
            else:
                k = np.concatenate([k, [float(ny)]])

        # Purge nearby peaks
        if 'm' in m:
            if len(v) > 0:
                iv = np.argmax(v)
                v = np.array([v[iv]])
                k = np.array([k[iv]])
        elif w is not None and np.isscalar(w) and w > 0:
            j = np.where(k[1:] - k[:-1] <= w)[0]
            while len(j) > 0:
                j = j + (v[j] >= v[j + 1]).astype(int)
                k = np.delete(k, j)
                v = np.delete(v, j)
                j = np.where(k[1:] - k[:-1] <= w)[0]
    elif ny > 0 and ('f' in m or 'l' in m):
        v = y.copy()
        if x is not None:
            k = x.copy()
        else:
            k = np.array([1.0])

    if 'v' in m:
        v = -v

    return k, v
