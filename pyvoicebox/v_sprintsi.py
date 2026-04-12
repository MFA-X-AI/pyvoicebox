"""V_SPRINTSI - Print value with SI multiplier."""

from __future__ import annotations
import numpy as np


def v_sprintsi(x, d=-3, w=0, u=' ') -> str:
    """Format a number with an SI multiplier prefix.

    Parameters
    ----------
    x : float
        Value to print.
    d : int, optional
        Decimal places (+ve) or significant digits (-ve). Default -3.
    w : int, optional
        Minimum total width. Default 0.
    u : str, optional
        Unit string. Default ' '.

    Returns
    -------
    s : str
        Formatted string.
    """
    f = 'qryzafpnum kMGTPEZYRQ'
    f0 = f.index(' ')
    emin = 3 - 3 * f0
    emax = 3 * (len(f) - f0)

    if x == 0:
        e = 0
    else:
        e = int(np.floor(np.log10(abs(x))))

    k = int(np.floor(max(emin, min(emax, e)) / 3))
    dp = max(0, d, 3 * k - d - e - 1)

    if w <= 0 and dp:
        w = abs(w)
        scaled = round(x * 10 ** (dp - 3 * k))
        remainder = scaled % (10 ** dp)
        # Find trailing zeros to eliminate
        for i in range(dp, 0, -1):
            if remainder % (10 ** i) == 0 and i < dp:
                continue
            else:
                break
        # Count how many trailing zeros
        trail = 0
        for i in range(dp, 0, -1):
            if int(remainder) % (10 ** i) == 0:
                trail = i
                break
        dp = dp - trail

    scaled_val = x * 10 ** (-3 * k)
    prefix = f[k + f0] if k != 0 else ''

    if u and u[0] == ' ':
        if k != 0:
            num_str = f'{scaled_val:{max(w - 2, 0)}.{dp}f}'
            s = f'{num_str} {prefix}{u[1:]}'
        else:
            num_str = f'{scaled_val:{max(w - 1, 0)}.{dp}f}'
            s = f'{num_str} {u[1:]}'
    else:
        if k != 0:
            num_str = f'{scaled_val:{max(w - 2, 0)}.{dp}f}'
            s = f'{num_str}{prefix}{u}'
        else:
            num_str = f'{scaled_val:{max(w - 1, 0)}.{dp}f}'
            s = f'{num_str}{u}'

    return s
