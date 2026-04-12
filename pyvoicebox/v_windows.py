"""V_WINDOWS - Generate a standard windowing function."""

from __future__ import annotations
import numpy as np
from scipy.special import i0 as besseli0


_WNAM = {
    'rectangle': 1, 'r': 1,
    'hann': 2, 'hanning': 2, 'vonhann': 2, 'n': 2,
    'hamming': 3, 'm': 3,
    'harris3': 4, '3': 4,
    'harris4': 5, '4': 5,
    'blackman': 6, 'b': 6,
    'vorbis': 7, 'v': 7,
    'rsqvorbis': 8, 'w': 8,
    'triangle': 9, 't': 9,
    'cos': 10, 'c': 10,
    'kaiser': 11, 'k': 11,
    'gaussian': 12, 'g': 12,
    'cauchy': 13, 'y': 13,
    'dolph': 14, 'd': 14,
    'tukey': 15, 'u': 15,
}

# Windows whose endpoints are zero
_ZERO_END_WINDOWS = {2, 6, 7, 9, 10, 15}

# kk table for sample positioning
_KK = np.array([
    [-1, 1, 1, -1], [0, 0, 2, -2], [0, 1, 2, -1],    # w, h, c (normal)
    [-1, 0, 1, 0],  [0, 0, 2, 0],  [0, 1, 2, 1],      # lw, lh, lc
    [-1, 2, 1, 0],  [0, 0, 2, -2], [0, 1, 2, -1],      # rw, rh, rc
    [-1, 1, 1, -1], [0, 0, 2, -2], [0, 1, 2, -1],      # bw, bh, bc
    [-1, 1, 1, 1],  [0, 0, 2, 0],  [0, 1, 2, 1],       # nw, nh, nc
    [-1, 1, 1, 0],  [0, 0, 2, -1], [0, 1, 2, 0],       # sw, sh, sc
])


def v_windows(wtype, n=256, mode=None, p=None, ov=None) -> np.ndarray:
    """Generate a standard windowing function.

    Parameters
    ----------
    wtype : str or int
        Window type name or code.
    n : int, optional
        Number of output points. Default 256.
    mode : str, optional
        Options string controlling scaling and sampling.
    p : float or array_like, optional
        Parameter(s) for parameterized windows.
    ov : int, optional
        Overlap for convolution with rectangle ('o' option).

    Returns
    -------
    w : ndarray
        Window values (1D array of length n).
    """
    # Resolve window type
    if isinstance(wtype, str):
        wtype_int = _WNAM.get(wtype.lower())
        if wtype_int is None:
            raise ValueError(f'Unknown window type: {wtype}')
    else:
        wtype_int = int(wtype)

    n = int(np.floor(n))

    if mode is None:
        mode = 'uw'

    # Parse mode flags
    mm = np.zeros(len(mode), dtype=int)
    ll = 'hc lrbns'
    for i, c in enumerate(ll):
        if c != ' ':
            for j, mc in enumerate(mode):
                if mc == c:
                    mm[j] = i - 2  # offset so h=-2, c=-1, l=1, r=2, b=3, n=4, s=5

    max_mm = max(mm) if len(mm) > 0 else 0
    min_mm = min(mm) if len(mm) > 0 else 0

    # k index into kk table (0-based)
    k = 3 * max(max_mm, 0) + max(-min_mm, 0)
    if k < 3:
        if wtype_int in _ZERO_END_WINDOWS:
            k += 12  # default to 'n' option

    kk_row = _KK[k]

    # Handle 'o' option (convolve with rectangle)
    do_conv = 'o' in mode
    if do_conv:
        if ov is None:
            ov = n // 2
        n = n - ov + 1
    else:
        ov = 0

    fn = int(np.floor(n))
    kp = kk_row[2] * n + kk_row[3]
    ks = kk_row[0] * fn + kk_row[1]
    v = ((np.arange(0, 2 * fn, 2) + ks) / kp)

    # Generate window
    if wtype_int == 1:  # rectangle
        w = np.ones_like(v)
    elif wtype_int == 2:  # hanning
        w = 0.5 + 0.5 * np.cos(np.pi * v)
    elif wtype_int == 3:  # hamming
        w = 0.54 + 0.46 * np.cos(np.pi * v)
    elif wtype_int == 4:  # harris3
        w = 0.42323 + 0.49755 * np.cos(np.pi * v) + 0.07922 * np.cos(2 * np.pi * v)
    elif wtype_int == 5:  # harris4
        w = (0.35875 + 0.48829 * np.cos(np.pi * v) +
             0.14128 * np.cos(2 * np.pi * v) + 0.01168 * np.cos(3 * np.pi * v))
    elif wtype_int == 6:  # blackman
        w = 0.42 + 0.5 * np.cos(np.pi * v) + 0.08 * np.cos(2 * np.pi * v)
    elif wtype_int == 7:  # vorbis
        w = np.sin(0.25 * np.pi * (1 + np.cos(np.pi * v)))
    elif wtype_int == 8:  # rsqvorbis
        w = 0.571 - 0.429 * np.cos(0.5 * np.pi * (1 + np.cos(np.pi * v)))
    elif wtype_int == 9:  # triangle
        pp = p if p is not None else 1
        if not np.isscalar(pp):
            pp = pp[0]
        w = 1 - np.abs(v) ** pp
    elif wtype_int == 10:  # cos
        pp = p if p is not None else 1
        if not np.isscalar(pp):
            pp = pp[0]
        w = np.cos(0.5 * np.pi * v) ** pp
    elif wtype_int == 11:  # kaiser
        pp = p if p is not None else 8
        if not np.isscalar(pp):
            pp = pp[0]
        w = besseli0(pp * np.sqrt(np.maximum(1 - v ** 2, 0))) / besseli0(pp)
    elif wtype_int == 12:  # gaussian
        pp = p if p is not None else 3
        if not np.isscalar(pp):
            pp = pp[0]
        w = np.exp(-0.5 * pp ** 2 * v ** 2)
    elif wtype_int == 13:  # cauchy
        pp = p if p is not None else 1
        if not np.isscalar(pp):
            pp = pp[0]
        w = (1 + (pp * v) ** 2) ** -1
    elif wtype_int == 14:  # dolph
        raise NotImplementedError('Dolph-Chebyshev window not yet implemented')
    elif wtype_int == 15:  # tukey
        pp = p if p is not None else 0.5
        if not np.isscalar(pp):
            pp = pp[0]
        if pp > 0:
            pp = min(pp, 1.0)
            w = 0.5 + 0.5 * np.cos(np.pi * np.maximum(1 + (np.abs(v) - 1) / pp, 0))
        else:
            w = np.ones_like(v)
    else:
        raise ValueError(f'Unknown window type code: {wtype_int}')

    # Convolve with rectangle
    if do_conv and ov > 0:
        w = np.cumsum(w)
        orig_n = n
        w_ext = np.zeros(orig_n + ov - 1)
        w_ext[:orig_n] = w
        w_ext[orig_n:orig_n + ov - 1] = w[orig_n - 1] - w[orig_n - ov:orig_n - 1]
        w_ext[ov:orig_n] = w_ext[ov:orig_n] - w[:orig_n - ov]
        w = w_ext
        n = orig_n + ov - 1

    # Scale
    g = 1.0
    for c in mode:
        if '1' <= c <= '9':
            g = 1.0 / (ord(c) - ord('0'))
            break

    if 'd' in mode:
        w = w * (g / np.sum(w))
    elif 'D' in mode or 'a' in mode:
        w = w * (g / np.mean(w))
    elif 'e' in mode.replace('ne', '').replace('se', ''):
        # 'e' for energy, but not 'ne' or 'se' mode letters
        if any(c == 'e' for c in mode if c not in 'ns'):
            w = w * np.sqrt(g / np.sum(w ** 2))
    elif 'E' in mode:
        w = w * np.sqrt(g / np.mean(w ** 2))
    elif 'p' in mode.replace('sp', ''):
        w = w * (g / np.max(w))

    if 'q' in mode:
        w = np.sqrt(w)

    return w
