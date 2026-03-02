"""V_ZOOMFFT - DTFT evaluated over a linear frequency range."""

import numpy as np


def v_zoomfft(x, n=None, m=None, s=0, d=None):
    """DTFT evaluated over a linear frequency range Y=(X,N,M,S,D).

    Parameters
    ----------
    x : array_like
        Input vector or array.
    n : float, optional
        Reciprocal of normalized frequency increment (can be non-integer).
        The frequency increment is fs/n. Default: size(x, d).
    m : int, optional
        Number of output points is floor(m). Default: floor(n).
    s : float, optional
        Starting frequency index (can be non-integer).
        The starting frequency is s*fs/n. Default: 0.
    d : int, optional
        Axis along which to do FFT (0-based). Default: first non-singleton.

    Returns
    -------
    y : ndarray
        Output DTFT coefficients.
    f : ndarray
        Normalized frequencies (1 corresponds to fs).
    """
    x = np.asarray(x, dtype=complex)
    e = list(x.shape)
    p = x.size

    if d is None:
        dims = [i for i, ei in enumerate(e) if ei > 1]
        if dims:
            d = dims[0]
        else:
            d = 0

    k_len = e[d]
    q = p // k_len

    if d == 0:
        z = x.reshape(k_len, q)
    else:
        z = np.moveaxis(x, d, 0).reshape(k_len, q)

    if n is None:
        n = k_len
    if m is None:
        m = int(np.floor(n))
    else:
        m = int(np.floor(m))
    if s is None:
        s = 0

    k = k_len

    l = int(2 ** np.ceil(np.log2(m + k - 1)))  # next power of 2

    if n == int(n) and s == int(s) and int(n) < 2 * l and int(n) >= k:
        # Quickest to do a normal FFT
        ni = int(n)
        a = np.fft.fft(z, n=ni, axis=0)
        si = int(s)
        indices = np.mod(np.arange(si, si + m), ni)
        y = a[indices, :]
    else:
        # Chirp z-transform (Bluestein's algorithm)
        b = np.exp(1j * np.pi * np.mod((s + np.arange(1 - k, m)) ** 2, 2 * n) / n)
        c = np.conj(b[k - 1:k - 1 + m])
        h = np.fft.fft(b, n=l)
        g = np.exp(-1j * np.pi * np.mod(np.arange(k) ** 2, 2 * n) / n)
        a = np.fft.ifft(np.fft.fft(z * g[:, np.newaxis], n=l, axis=0) * h[:, np.newaxis], axis=0)
        y = a[k - 1:k - 1 + m, :] * c[:, np.newaxis]

    if d == 0:
        e_out = list(e)
        e_out[d] = m
        y = y.reshape(e_out)
    else:
        moved_shape = list(np.moveaxis(x, d, 0).shape)
        moved_shape[0] = m
        y = y.reshape(moved_shape)
        y = np.moveaxis(y, 0, d)

    f = (s + np.arange(m)) / n

    return y, f
