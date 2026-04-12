"""V_RSFFT - FFT of a real symmetric spectrum."""

from __future__ import annotations
import numpy as np


def v_rsfft(y, n=None) -> np.ndarray:
    """FFT of a real symmetric spectrum X=(Y,N).

    Y is the "first half" of a symmetric real input signal and X is the
    "first half" of the symmetric real Fourier transform.

    If N is even, the "first half" contains 1+N/2 elements.
    If N is odd, the "first half" contains (N+1)/2 elements.
    If N is omitted it will be taken to be 2*(length(Y)-1) and is always even.

    If Y is a matrix (2-D), the transform is performed along each column.

    The inverse function is y = v_rsfft(x, n) / n.

    Parameters
    ----------
    y : array_like
        Real input data. Must be real-valued.
    n : int, optional
        Full signal length. Default: 2*(M-1) where M is the number of rows.

    Returns
    -------
    x : ndarray
        The first half of the symmetric real Fourier transform.
    """
    y = np.asarray(y, dtype=float)
    if not np.all(np.isreal(y)):
        raise ValueError('RSFFT: Input must be real')

    fl = False
    if y.ndim == 1:
        fl = True
        y = y[:, np.newaxis]
    elif y.shape[0] == 1:
        fl = True
        y = y.T

    m, k = y.shape

    if n is None:
        n = 2 * m - 2
    else:
        mm = 1 + n // 2
        if mm > m:
            y = np.vstack([y, np.zeros((mm - m, k))])
        elif mm < m:
            y = y[:mm, :]
        m = mm

    # Build full symmetric signal and take FFT
    # MATLAB: fft([y; y(n-m+1:-1:2, :)])
    # n-m+1 in 1-based -> n-m in 0-based, down to index 2 in 1-based -> index 1 in 0-based
    full = np.vstack([y, y[n - m:0:-1, :]])
    x = np.real(np.fft.fft(full, axis=0))
    x = x[:m, :]  # keep first m rows

    if fl:
        x = x.ravel()

    return x
