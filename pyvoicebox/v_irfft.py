"""V_IRFFT - Inverse FFT of a conjugate symmetric spectrum."""

from __future__ import annotations
import numpy as np


def v_irfft(y, n=None, d=None) -> np.ndarray:
    """Inverse FFT of a conjugate symmetric spectrum X=(Y,N,D).

    Parameters
    ----------
    y : array_like
        The first half of a complex spectrum (as produced by v_rfft).
    n : int, optional
        Number of output points to generate. Default: 2*M-2 where M is size
        of y along axis d. Note: default is always even; specify n explicitly
        if odd output is desired.
    d : int, optional
        Axis along which to perform the transform. Default: first non-singleton
        dimension.

    Returns
    -------
    x : ndarray
        Real inverse DFT of y.
    """
    y = np.asarray(y, dtype=complex)
    s = list(y.shape)
    ns = len(s)

    # Scalar case
    if y.size == 1:
        return np.real(y).copy()

    if d is None:
        for i, si in enumerate(s):
            if si > 1:
                d = i
                break
        if d is None:
            d = 0

    m = s[d]
    k = y.size // m  # number of FFTs to do

    # Reshape: move dimension d to the front, then reshape to (m, k)
    if d == 0:
        v = y.reshape(m, k)
    else:
        perm = list(range(d, ns)) + list(range(0, d))
        v = np.transpose(y, perm).reshape(m, k)

    if n is None:
        n = 2 * m - 2  # default output length
    else:
        mm = 1 + n // 2  # expected input length
        if mm > m:
            v = np.vstack([v, np.zeros((mm - m, k), dtype=complex)])
        elif mm < m:
            v = v[:mm, :]
        m = mm

    v[0, :] = np.real(v[0, :])  # force DC element real

    if n % 2:  # odd output length
        full = np.vstack([v, np.conj(v[m - 1:0:-1, :])])
        x = np.real(np.fft.ifft(full, axis=0))
    else:  # even output length
        v[m - 1, :] = np.real(v[m - 1, :])  # force Nyquist element real
        w = np.ones((1, k))
        t = -0.5j * np.exp((2j * np.pi / n) * np.arange(m))
        t = t[:, np.newaxis]
        z = (t + 0.5) * (np.conj(v[::-1, :]) - v) + v
        z = z[:m - 1, :]  # remove last row (z[m-1,:])
        zz = np.fft.ifft(z, axis=0)
        x = np.zeros((n, k))
        x[0::2, :] = np.real(zz)
        x[1::2, :] = np.imag(zz)

    s[d] = n  # change output dimension
    if d == 0:
        x = x.reshape(s)
    else:
        # Reorder dimensions: s was permuted as [d:ns, 0:d]
        perm_shape = [s[i] for i in (list(range(d, ns)) + list(range(0, d)))]
        x = x.reshape(perm_shape)
        # Inverse permutation
        inv_perm = list(range(ns + 1 - d, ns)) + list(range(0, ns + 1 - d))
        x = np.transpose(x, inv_perm)

    return x
