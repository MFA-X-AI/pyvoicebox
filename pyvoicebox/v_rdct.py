"""V_RDCT - Discrete cosine transform of real data."""

from __future__ import annotations
import numpy as np


def v_rdct(x, n=None, a=None, b=1.0) -> np.ndarray:
    """Discrete cosine transform of real data Y=(X,N,A,B).

    Parameters
    ----------
    x : array_like
        Real-valued input data. Transform is applied to columns.
    n : int, optional
        Transform length. x will be zero-padded or truncated to length n.
        Default: number of rows of x.
    a : float, optional
        Output scale factor. Default: sqrt(2*n).
    b : float, optional
        Output scale factor for DC term. Default: 1.

    Returns
    -------
    y : ndarray
        DCT output data.
    """
    x = np.asarray(x, dtype=float)

    fl = False
    if x.ndim == 1:
        fl = True
        x = x[:, np.newaxis]
    elif x.shape[0] == 1:
        fl = True
        x = x.T

    m, k = x.shape
    if n is None:
        n = m
    if a is None:
        a = np.sqrt(2.0 * n)

    # Zero-pad or truncate
    if n > m:
        x = np.vstack([x, np.zeros((n - m, k))])
    elif n < m:
        x = x[:n, :]

    # Reorder: [x(1:2:n,:); x(2*fix(n/2):-2:2,:)]
    # MATLAB 1-based: odd indices 1,3,5,... then even indices from 2*fix(n/2) down by 2 to 2
    odd_idx = np.arange(0, n, 2)  # 0-based: 0,2,4,...
    even_top = 2 * (n // 2) - 1   # 0-based version of 2*fix(n/2)
    even_idx = np.arange(even_top, 0, -2)  # 0-based: ...,3,1
    idx = np.concatenate([odd_idx, even_idx])
    x = x[idx, :]

    # Compute z multipliers
    # MATLAB: z = [sqrt(2) 2*exp((-0.5i*pi/n)*(1:n-1))].';
    z = np.zeros(n, dtype=complex)
    z[0] = np.sqrt(2.0)
    z[1:] = 2.0 * np.exp((-0.5j * np.pi / n) * np.arange(1, n))

    # y = real(fft(x) .* z(:, ones(1,k))) / a
    y = np.real(np.fft.fft(x, axis=0) * z[:, np.newaxis]) / a
    y[0, :] = y[0, :] * b

    if fl:
        y = y.ravel()

    return y
