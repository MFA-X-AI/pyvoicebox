"""V_STDSPECTRUM - Generate standard acoustic/speech spectra (simplified)."""

from __future__ import annotations
import numpy as np
from scipy.signal import bilinear, freqz


def v_stdspectrum(s, m='s', f=8192, n=None, zi=None, bs=None, as_=None) -> tuple[np.ndarray, np.ndarray]:
    """Generate standard acoustic/speech spectra in s- or z-domain.

    Simplified implementation supporting s-domain output and basic z-domain
    conversion for the most common spectrum types.

    Parameters
    ----------
    s : int or str
        Spectrum type:
            1 = White
            2 = A-Weight
            3 = B-Weight
            4 = C-Weight
            9 = USASI
    m : str, optional
        Mode: 's' for s-domain, 'z' for z-domain. Default 's'.
    f : float, optional
        Sample frequency (for 'z' mode). Default 8192.
    n : int, optional
        Number of samples (for 't' mode).

    Returns
    -------
    b : ndarray
        Numerator coefficients.
    a : ndarray
        Denominator coefficients.
    """
    # S-domain spectrum definitions (zeros, poles, gain)
    spectra = _get_spectra()

    if isinstance(s, str):
        names = {name.lower(): idx for idx, name in enumerate(spectra.keys())}
        si = names.get(s.lower(), 0)
        sn = s
    else:
        si = int(s)

    spec_list = list(spectra.values())
    if si < 1 or si > len(spec_list):
        if si == 0 and bs is not None:
            sb = np.asarray(bs, dtype=float)
            sa = np.asarray(as_, dtype=float) if as_ is not None else np.array([1.0])
        else:
            raise ValueError(f'Undefined spectrum type: {s}')
    else:
        sb, sa = spec_list[si - 1]

    m1 = m[0] if m else 's'

    if m1 == 's':
        return sb, sa
    elif m1 == 'z':
        if si == 1:  # White noise
            return np.array([1.0]), np.array([1.0])
        # Use bilinear transform
        bz, az = bilinear(sb, sa, f)
        return bz, az
    else:
        return sb, sa


def _get_spectra():
    """Get s-domain transfer functions for standard spectra."""
    spectra = {}

    # 1: White noise
    spectra['white'] = (np.array([1.0]), np.array([1.0]))

    # 2: A-weighting
    # Standard A-weighting poles and zeros
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997  # gain at 1000 Hz
    pi = np.pi

    NUM = np.array([(2 * pi * f4) ** 2 * (10 ** (A1000 / 20))])
    sz = np.array([0, 0, 0, 0])
    sp = np.array([
        -2 * pi * f1, -2 * pi * f1,
        -2 * pi * f2, -2 * pi * f3,
        -2 * pi * f4, -2 * pi * f4
    ])
    # Build transfer function
    sb = np.real(np.poly(sz)) * (2 * pi * f4) ** 2 * 10 ** (A1000 / 20)
    sa = np.real(np.poly(sp))
    # Normalize gain at 1000 Hz
    h1000 = np.polyval(sb, 2j * pi * 1000) / np.polyval(sa, 2j * pi * 1000)
    sb = sb / abs(h1000)
    spectra['a-weight'] = (sb, sa)

    # 3: B-weighting (simplified)
    sz_b = np.array([0, 0, 0])
    sp_b = np.array([
        -2 * pi * f1, -2 * pi * f1,
        -2 * pi * 158.489,
        -2 * pi * f4, -2 * pi * f4
    ])
    sb_b = np.real(np.poly(sz_b))
    sa_b = np.real(np.poly(sp_b))
    h1000_b = np.polyval(sb_b, 2j * pi * 1000) / np.polyval(sa_b, 2j * pi * 1000)
    sb_b = sb_b / abs(h1000_b)
    spectra['b-weight'] = (sb_b, sa_b)

    # 4: C-weighting
    sz_c = np.array([0, 0])
    sp_c = np.array([
        -2 * pi * f1, -2 * pi * f1,
        -2 * pi * f4, -2 * pi * f4
    ])
    sb_c = np.real(np.poly(sz_c))
    sa_c = np.real(np.poly(sp_c))
    h1000_c = np.polyval(sb_c, 2j * pi * 1000) / np.polyval(sa_c, 2j * pi * 1000)
    sb_c = sb_c / abs(h1000_c)
    spectra['c-weight'] = (sb_c, sa_c)

    # Placeholders for other types
    for i in range(5, 16):
        spectra[f'type{i}'] = (np.array([1.0]), np.array([1.0]))

    # 9: USASI - simple model
    spectra['usasi'] = (np.array([72.65, 0, -72.65]),
                        np.real(np.poly(np.exp(-np.array([100, 320]) * 2 * np.pi / 8000) * 8000)))

    return spectra
