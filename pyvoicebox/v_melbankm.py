"""V_MELBANKM - Determine matrix for a mel/erb/bark-spaced filterbank."""

from __future__ import annotations
from typing import Any
import numpy as np
import scipy.sparse as sp
from .v_frq2mel import v_frq2mel
from .v_mel2frq import v_mel2frq
from .v_frq2bark import v_frq2bark
from .v_bark2frq import v_bark2frq
from .v_frq2erb import v_frq2erb
from .v_erb2frq import v_erb2frq


def v_melbankm(p=None, n=256, fs=11025, fl=0, fh=0.5, w='tz') -> tuple[Any, np.ndarray, int, int]:
    """Determine matrix for a mel/erb/bark-spaced filterbank.

    Parameters
    ----------
    p : int or float, optional
        Number of filters or filter spacing in k-mel/bark/erb.
        Default: ceil(4.6*log10(fs)).
    n : int
        Length of FFT.
    fs : float
        Sample rate in Hz.
    fl : float
        Low end of lowest filter as fraction of fs (or Hz if 'h'/'H' in w).
    fh : float
        High end of highest filter as fraction of fs.
    w : str
        Options string (see MATLAB docs).

    Returns
    -------
    x : scipy.sparse matrix
        Filterbank matrix (p, 1+floor(n/2)) or (p, mx-mn+1).
    mc : ndarray
        Filterbank centre frequencies in mel/erb/bark.
    mn : int
        Lowest FFT bin with non-zero coefficient.
    mx : int
        Highest FFT bin with non-zero coefficient.
    """
    sfact = 2 - int('s' in w)  # 1 if single sided else 2

    # Determine warping
    wr = ' '
    for ch in w:
        if ch in 'lebf':
            wr = ch

    if 'h' in w or 'H' in w:
        mflh = np.array([fl, fh], dtype=float)
    else:
        mflh = np.array([fl * fs, fh * fs], dtype=float)

    if 'H' not in w:
        if wr == 'f':
            pass
        elif wr == 'l':
            if fl <= 0:
                raise ValueError("Low frequency limit must be >0 for 'l' option")
            mflh = np.log10(mflh)
        elif wr == 'e':
            mflh = v_frq2erb(mflh)[0]
        elif wr == 'b':
            mflh = v_frq2bark(mflh)[0]
        else:
            mflh = v_frq2mel(mflh)[0]

    melrng = mflh[1] - mflh[0]
    fn2 = n // 2

    if p is None:
        p = int(np.ceil(4.6 * np.log10(fs)))

    if 'c' in w:
        if p < 1:
            p = int(round(melrng / (p * 1000))) + 1
        melinc = melrng / (p - 1)
        mflh = mflh + np.array([-1, 1]) * melinc
    else:
        if p < 1:
            p = int(round(melrng / (p * 1000))) - 1
        melinc = melrng / (p + 1)

    # Calculate FFT bins for filter boundaries
    edges = mflh[0] + np.array([0, 1, p, p + 1]) * melinc
    if wr == 'f':
        blim = edges * n / fs
    elif wr == 'l':
        blim = 10.0 ** edges * n / fs
    elif wr == 'e':
        blim = v_erb2frq(edges)[0] * n / fs
    elif wr == 'b':
        blim = v_bark2frq(edges)[0] * n / fs
    else:
        blim = v_mel2frq(edges)[0] * n / fs

    mc = mflh[0] + (np.arange(1, p + 1)) * melinc  # mel centre frequencies
    b1 = int(np.floor(blim[0])) + 1  # lowest FFT bin_0 required
    b4 = min(fn2, int(np.ceil(blim[3])) - 1)  # highest FFT bin_0 required

    # Map FFT bins to filter centres
    bins = np.arange(b1, b4 + 1)
    freqs = bins * fs / n
    if wr == 'f':
        pf = (freqs - mflh[0]) / melinc
    elif wr == 'l':
        pf = (np.log10(freqs) - mflh[0]) / melinc
    elif wr == 'e':
        pf = (v_frq2erb(freqs)[0] - mflh[0]) / melinc
    elif wr == 'b':
        pf = (v_frq2bark(freqs)[0] - mflh[0]) / melinc
    else:
        pf = (v_frq2mel(freqs)[0] - mflh[0]) / melinc

    # Remove incorrect entries due to rounding
    if len(pf) > 0 and pf[0] < 0:
        pf = pf[1:]
        b1 = b1 + 1
    if len(pf) > 0 and pf[-1] >= p + 1:
        pf = pf[:-1]
        b4 = b4 - 1

    fp = np.floor(pf).astype(int)
    pm = pf - fp

    k2_arr = np.where(fp > 0)[0]
    k2 = k2_arr[0] if len(k2_arr) > 0 else len(fp)
    k3_arr = np.where(fp < p)[0]
    k3 = k3_arr[-1] if len(k3_arr) > 0 else -1
    k4 = len(fp) - 1

    if 'y' in w:
        mn = 1
        mx = fn2 + 1
        # Build sparse matrix with preserved power
        r_list = []
        c_list = []
        v_list = []
        # Left edge: bins before k2
        for i in range(k2 + b1):
            r_list.append(0)
            c_list.append(i)
            v_list.append(1.0)
        # Middle: upper filters
        for i in range(k2, k3 + 1):
            r_list.append(fp[i])
            c_list.append(i + b1)
            v_list.append(pm[i])
        # Middle: lower filters
        for i in range(k2, k3 + 1):
            r_list.append(fp[i] - 1)
            c_list.append(i + b1)
            v_list.append(1.0 - pm[i])
        # Right edge: bins after k3
        for i in range(k3 + b1 + 1, fn2 + 1):
            r_list.append(p - 1)
            c_list.append(i)
            v_list.append(1.0)
        r_arr = np.array(r_list)
        c_arr = np.array(c_list)
        v_arr = np.array(v_list, dtype=float)
    else:
        # Standard case
        r1 = fp[:k3 + 1]  # filter number (0-based)
        c1 = np.arange(k3 + 1)
        v1 = pm[:k3 + 1]

        r2 = fp[k2:k4 + 1] - 1  # filter number (0-based)
        c2 = np.arange(k2, k4 + 1)
        v2 = 1.0 - pm[k2:k4 + 1]

        r_arr = np.concatenate([r1, r2])
        c_arr = np.concatenate([c1, c2])
        v_arr = np.concatenate([v1, v2])

        mn = b1 + 1  # lowest FFT bin_1
        mx = b4 + 1  # highest FFT bin_1

    # Handle negative frequencies
    if b1 < 0:
        c_arr = np.abs(c_arr + b1 - 1) - b1 + 1

    # Apply window shape
    if 'n' in w:
        v_arr = 0.5 - 0.5 * np.cos(v_arr * np.pi)
    elif 'm' in w:
        v_arr = 0.5 - 0.46 / 1.08 * np.cos(v_arr * np.pi)

    # Double all except DC and Nyquist
    if sfact == 2:
        msk = (c_arr + mn > 2) & (c_arr + mn < n - fn2 + 2)
        v_arr[msk] = 2.0 * v_arr[msk]

    # Build sparse matrix
    # Convert to 0-based indices for the full matrix
    x = sp.csr_matrix((v_arr, (r_arr, c_arr + mn - 1)),
                       shape=(p, 1 + fn2))

    if 'u' in w:
        sx = np.array(x.sum(axis=1)).ravel()
        sx[sx == 0] = 1.0
        x = sp.diags(1.0 / sx) @ x

    return x, mc, mn, mx
