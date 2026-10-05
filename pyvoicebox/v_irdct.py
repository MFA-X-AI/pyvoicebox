"""V_IRDCT - Inverse discrete cosine transform of real data."""

from __future__ import annotations
import numpy as np


def v_irdct(y, n=None, a=None, b=1.0) -> np.ndarray:
    """Inverse discrete cosine transform of real data X=(Y,N,A,B).

    Parameters
    ----------
    y : array_like
        DCT coefficients (as produced by v_rdct). Transform applied to columns.
    n : int, optional
        Output length. Default: number of rows of y.
    a : float, optional
        Scale factor. Default: sqrt(2*m) where m is the number of rows of y.
    b : float, optional
        DC scale factor. Default: 1.

    Returns
    -------
    x : ndarray
        Inverse DCT output data.
    """
    y = np.asarray(y, dtype=float)

    fl = False
    if y.ndim == 1:
        fl = True
        y = y[:, np.newaxis]
    elif y.shape[0] == 1:
        fl = True
        y = y.T

    m_orig, k = y.shape

    if a is None:
        a = np.sqrt(2.0 * m_orig)
    if n is None:
        n = m_orig

    # Zero-pad or truncate
    if n > m_orig:
        y = np.vstack([y, np.zeros((n - m_orig, k))])
    elif n < m_orig:
        y = y[:n, :]

    x = np.zeros((n, k))
    m = (n + 1) // 2  # fix((n+1)/2)
    p = n - m

    # z = 0.5*exp((0.5i*pi/n)*(1:p)).'
    z = 0.5 * np.exp((0.5j * np.pi / n) * np.arange(1, p + 1))
    z = z[:, np.newaxis]

    # u = (y(2:p+1,:) - 1i*y(n:-1:m+1,:)) .* z(:,w) * a
    # MATLAB 1-based: y(2:p+1,:) -> 0-based: y[1:p+1,:]
    # MATLAB 1-based: y(n:-1:m+1,:) -> 0-based: y[n-1:m:-1,:]  (but n is already length, so y has indices 0..n-1)
    u = (y[1:p + 1, :] - 1j * y[n - 1:m - 1:-1, :]) * z * a

    # y = [y(1,:)*sqrt(0.5)*a/b; u(1:m-1,:)]
    yy = np.vstack([y[0:1, :] * np.sqrt(0.5) * a / b, u[:m - 1, :]])

    if m == p:
        # Even n case
        # z = -0.5i*exp((2i*pi/n)*(0:m-1)).'
        z2 = -0.5j * np.exp((2j * np.pi / n) * np.arange(m))
        z2 = z2[:, np.newaxis]
        # y = (z(:,w)+0.5).*(conj(flipud(u))-y)+y
        yy = (z2 + 0.5) * (np.conj(u[::-1, :]) - yy) + yy
        # z = ifft(y,[],1)
        zz = np.fft.ifft(yy, axis=0)
        uu = np.real(zz)
        yi = np.imag(zz)
        # Interleave the real and imaginary halves. When m is odd, the
        # first two strides contain one more sample than the last two.
        upper = (m + 1) // 2
        lower = m // 2
        x[0::4, :] = uu[:upper, :]
        x[1::4, :] = yi[::-1, :][:upper, :]
        x[2::4, :] = yi[:lower, :]
        x[3::4, :] = uu[::-1, :][:lower, :]
    else:
        # Odd n case
        # z = real(ifft([y; conj(flipud(u))]))
        full = np.vstack([yy, np.conj(u[::-1, :])])
        zz = np.real(np.fft.ifft(full, axis=0))
        x[0::2, :] = zz[:m, :]
        x[1::2, :] = zz[n - 1:m - 1:-1, :]

    if fl:
        x = x.ravel()

    return x
