"""V_IMPORTSII - Calculate the SII importance function."""

import numpy as np
from .v_frq2bark import v_frq2bark


# Precomputed constants
_ci4 = 0.0783
_ci18 = 0.8861
_mi = (_ci18 - _ci4) / 14.0
_ci = _ci4 - 4.0 * _mi
_ai = _mi**2 / (4.0 * (4.0 * _mi + _ci))
_bi = _mi**2 / (4.0 * (1.0 - 18.0 * _mi - _ci))
_xi0 = 4.0 - _mi / (2.0 * _ai)
_xi1 = 18.0 + _mi / (2.0 * _bi)


def v_importsii(f, m=''):
    """Calculate the SII importance function per Hz or per Bark.

    Parameters
    ----------
    f : array_like
        Frequencies in Hz (or Bark if 'b' flag).
    m : str, optional
        Mode string:
          'b' : Frequencies given in Bark rather than Hz.
          'c' : Calculate cumulative importance.
          'd' : Calculate importance of n-1 bands.
          'h' : Calculate importance per Hz or per Bark.

    Returns
    -------
    q : ndarray
        Importance values.

    References
    ----------
    [1] ANSI Standard S3.5-1997 (R2007).
    [2] C. V. Pavlovic. JASA, 82:413-422, 1987.
    """
    f = np.asarray(f, dtype=float)

    if 'b' in m:
        b = f.copy()
        d_bark = np.ones_like(f)  # not needed for bark input
    else:
        b, d_bark = v_frq2bark(f)

    if 'c' in m or 'd' in m:
        q = _mi * b + _ci + _ai * (b < 4) * (b - 4)**2 - _bi * (b > 18) * (b - 18)**2
        q[b < _xi0] = 0.0
        q[b > _xi1] = 1.0
        if 'd' in m:
            q = q[1:] - q[:-1]
    else:
        q = _mi + _ai * (b < 4) * (b - 4) - _bi * (b > 18) * (b - 18)
        q[b < _xi0] = 0.0
        q[b > _xi1] = 0.0
        if 'b' not in m:
            q = q / d_bark

    return q
