"""V_BITSPREC - Round values to a specified fixed or floating precision."""

import numpy as np


def v_bitsprec(x, n=None, mode='sne'):
    """Round values to a specified number of bits precision.

    Parameters
    ----------
    x : array_like
        Input values.
    n : int, optional
        Number of bits.
    mode : str, optional
        Mode string 'uvw' where:
        u: 's' = significant bits (default), 'f' = fixed point
        v: 'n' = nearest, 'p' = toward +inf, 'm' = toward -inf, 'z' = toward zero
        w: 'e' = even (default), 'o' = odd, 'a' = away from zero, 'p'/'m' = as v

    Returns
    -------
    y : ndarray
        Rounded values.
    """
    x = np.asarray(x, dtype=float)

    if mode[0] == 'f':
        e = np.zeros_like(x, dtype=int)
    else:
        # Decompose x = f * 2^e where 0.5 <= |f| < 1
        f, e = np.frexp(x)
        x = f
        e = e.astype(int)

    n = int(n) if n is not None else 0
    xn = np.ldexp(x, n)

    en = (e - n).astype(int)

    if mode[1] == 'p':
        y = np.ldexp(np.ceil(xn), en)
    elif mode[1] == 'm':
        y = np.ldexp(np.floor(xn), en)
    elif mode[1] == 'z':
        y = np.ldexp(np.fix(xn), en)
    else:  # 'n' - round to nearest
        w = mode[2] if len(mode) > 2 else 'e'
        if w == 'a':
            y = np.ldexp(np.round(xn), en)
        elif w == 'p':
            y = np.ldexp(np.floor(xn + 0.5), en)
        elif w == 'm':
            y = np.ldexp(np.ceil(xn - 0.5), en)
        elif w == 'e':
            z = np.ldexp(x, n - 1)
            y = np.ldexp(
                np.floor(xn + 0.5) - np.floor(z + 0.75) + np.ceil(z - 0.25),
                en
            )
        elif w == 'o':
            z = np.ldexp(x, n - 1)
            y = np.ldexp(
                np.ceil(xn - 0.5) + np.floor(z + 0.75) - np.ceil(z - 0.25),
                en
            )
        else:
            y = np.ldexp(np.round(xn), en)

    return y
