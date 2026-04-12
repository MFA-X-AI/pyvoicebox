"""V_FILTBANKM - General filterbank matrix (mel/bark/erb/linear)."""

from __future__ import annotations
from typing import Any
import numpy as np
from scipy.sparse import csr_matrix
from pyvoicebox.v_frq2mel import v_frq2mel
from pyvoicebox.v_mel2frq import v_mel2frq
from pyvoicebox.v_frq2erb import v_frq2erb
from pyvoicebox.v_erb2frq import v_erb2frq
from pyvoicebox.v_frq2bark import v_frq2bark
from pyvoicebox.v_bark2frq import v_bark2frq


def v_filtbankm(p, n, fs, fl=0, fh=None, w='') -> tuple[Any, np.ndarray, int, int]:
    """Determine matrix for a filterbank with various frequency scales.

    Simplified implementation supporting mel, bark, erb, linear scales.

    Parameters
    ----------
    p : int
        Number of filters in filterbank.
    n : int
        Length of DFT.
    fs : float
        Sample rate in Hz.
    fl : float, optional
        Low frequency edge in Hz. Default 0.
    fh : float, optional
        High frequency edge in Hz. Default fs/2.
    w : str, optional
        Options: 'm'=mel, 'b'=bark, 'e'=erb, 'f'=linear [default].

    Returns
    -------
    x : ndarray, shape (p, 1+n//2)
        Filterbank matrix.
    cf : ndarray, shape (p,)
        Filter center frequencies in Hz.
    """
    if fh is None:
        fh = fs / 2.0

    nf = 1 + n // 2  # number of frequency bins
    fax = np.arange(nf) * fs / n  # frequency axis

    # Choose frequency scale
    if 'm' in w:
        frq2scale = v_frq2mel
        scale2frq = v_mel2frq
    elif 'b' in w:
        frq2scale = v_frq2bark
        scale2frq = v_bark2frq
    elif 'e' in w:
        frq2scale = v_frq2erb
        scale2frq = v_erb2frq
    else:
        frq2scale = lambda x: x
        scale2frq = lambda x: x

    # Convert frequency limits to chosen scale
    fl_s = frq2scale(fl)
    fh_s = frq2scale(fh)

    # Create p+2 equally spaced points in the chosen scale (including edges)
    cf_s = np.linspace(fl_s, fh_s, p + 2)
    cf_hz = scale2frq(cf_s)

    # Center frequencies (excluding edges)
    cf = cf_hz[1:-1]

    # Build the filterbank matrix
    x = np.zeros((p, nf))
    for i in range(p):
        lo = cf_hz[i]
        mid = cf_hz[i + 1]
        hi = cf_hz[i + 2]

        # Rising slope
        mask_rise = (fax >= lo) & (fax <= mid)
        if mid > lo:
            x[i, mask_rise] = (fax[mask_rise] - lo) / (mid - lo)

        # Falling slope
        mask_fall = (fax >= mid) & (fax <= hi)
        if hi > mid:
            x[i, mask_fall] = (hi - fax[mask_fall]) / (hi - mid)

    return x, cf
