"""V_RANGELIM - Limit the range of matrix elements."""

import numpy as np


def v_rangelim(x, r, m='lp'):
    """Limit the range of matrix elements.

    Parameters
    ----------
    x : array_like
        Input data.
    r : float or array_like
        If 2-element: explicit [min, max] limits.
        If scalar: range specification depending on mode.
    m : str, optional
        Mode string:
          'd' - range r in dB: 20*log10(max/min)
          'r' - range r is max/min ratio
          'l' - range r is max-min difference (default)
          'p' - max(x) is top of range (default)
          't' - min(x) is bottom of range
          'g' - geometric mean is centre of range
          'u' - mean is centre of range
          'm' - median is centre of range
          'c' - clip out-of-range values (default)
          'n' - set out-of-range values to NaN

    Returns
    -------
    y : ndarray
        Output data, same shape as x.
    """
    x = np.asarray(x, dtype=float)
    r = np.atleast_1d(np.asarray(r, dtype=float))

    if len(r) > 1:
        p = r[0]
        q = r[1]
    else:
        r_val = r[0]

        # Determine reference type: g=1, u=2, m=3, t=4, p=5
        ref_map = {'g': 1, 'u': 2, 'm': 3, 't': 4}
        ref_id = 5  # default 'p'
        for char, val in ref_map.items():
            if char in m:
                ref_id = val
                break

        # Determine range type: d=1, r=2, l=3
        if 'd' in m:
            ir = 1
        elif 'r' in m:
            ir = 2
        else:
            ir = 3

        if ir == 1:  # dB
            r_val = 10 ** (0.05 * r_val)

        xflat = x.ravel()

        if ir == 3:  # linear range
            if ref_id == 5:  # 'p': peak
                q = np.max(xflat)
                p = q - r_val
            elif ref_id == 4:  # 't': trough
                p = np.min(xflat)
                q = p + r_val
            elif ref_id == 3:  # 'm': median
                p = np.median(xflat) - 0.5 * r_val
                q = p + r_val
            elif ref_id == 2:  # 'u': mean
                p = np.mean(xflat) - 0.5 * r_val
                q = p + r_val
            elif ref_id == 1:  # 'g': geometric mean
                p = np.exp(np.mean(np.log(xflat))) - 0.5 * r_val
                q = p + r_val
        else:  # ratio range ('r' or 'd')
            if ref_id == 5:  # 'p': peak
                q = np.max(xflat)
                p = q / r_val
            elif ref_id == 4:  # 't': trough
                p = np.min(xflat)
                q = p * r_val
            elif ref_id == 3:  # 'm': median
                p = np.median(xflat) / np.sqrt(r_val)
                q = p * r_val
            elif ref_id == 2:  # 'u': mean
                p = np.mean(xflat) / np.sqrt(r_val)
                q = p * r_val
            elif ref_id == 1:  # 'g': geometric mean
                p = np.exp(np.mean(np.log(xflat))) / np.sqrt(r_val)
                q = p * r_val

    y = x.copy()
    if 'n' in m:
        y[(x < p) | (x > q)] = np.nan
    else:
        y[x < p] = p
        y[x > q] = q

    return y
