"""V_IRDCT - Inverse discrete cosine transform of real data."""

import numpy as np


def v_irdct(y, n=None, a=None, b=1.0):
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
        q = m // 2
        h = (m % 2) / 2.0  # rem(m,2)/2

        # MATLAB uses 1-based indexing for x(1:4:n,:), x(2:4:n,:), etc.
        # In 0-based: x[0::4], x[1::4], x[2::4], x[3::4]
        # MATLAB: x(1:4:n,:)=u(1:q+h,:)   -> 0-based: x[0::4]=uu[0:int(q+h)]
        # MATLAB: x(2:4:n,:)=y(m:-1:q+1-h,:) -> 0-based: x[1::4]=yi[m-1:int(q-h):-1]
        # MATLAB: x(3:4:n,:)=y(1:q-h,:) -> 0-based: x[2::4]=yi[0:int(q-h)]
        # MATLAB: x(4:4:n,:)=u(m:-1:q+1+h,:) -> 0-based: x[3::4]=uu[m-1:int(q+h):-1]
        # Note: h is 0 when m is even, 0.5 when m is odd
        # But these are used as integer indices; in MATLAB q+h is floor/ceil depending
        ih = int(h)  # 0 if m even, but h=0.5 rounds... let's be careful

        # When m is even: h=0, q=m/2
        #   x(1:4:n)=u(1:q) -> x[0::4]=uu[0:q]
        #   x(2:4:n)=y(m:-1:q+1) -> x[1::4]=yi[m-1:q-1:-1]
        #   x(3:4:n)=y(1:q) -> x[2::4]=yi[0:q]
        #   x(4:4:n)=u(m:-1:q+1) -> x[3::4]=uu[m-1:q-1:-1]
        # When m is odd: h=0.5, q=(m-1)/2
        #   x(1:4:n)=u(1:q+0.5)=u(1:q+1) (since MATLAB rounds up for indexing)
        #   Actually in MATLAB, q+h where h=0.5 gives q+0.5, and 1:q+0.5 means 1:floor(q+0.5)
        #   For m odd, q=(m-1)/2, so q+0.5 = m/2 which is not integer... MATLAB truncates
        #   Actually MATLAB: 1:q+h where q=int, h=0.5 gives 1:q (since q+0.5 is not >= q+1)
        #   Wait, let me re-examine. In MATLAB, 1:2.5 = [1, 2], so it goes up to floor(2.5)=2.
        #   So: when m is odd, h=0.5:
        #     q = (m-1)/2  (integer since m is odd)
        #     q+h = q + 0.5 = m/2 (non-integer)
        #     1:q+h = 1:q  (MATLAB truncates to integer steps)
        #     q+1-h = q+0.5 (non-integer)
        #     m:-1:q+1-h = m:-1:q+1 (MATLAB ceil for reverse: goes down to ceil(q+0.5)=q+1)
        #     q-h = q-0.5 (non-integer)
        #     1:q-h = 1:q-1 (MATLAB floor)
        #     q+1+h = q+1.5
        #     m:-1:q+1+h = m:-1:q+2 (MATLAB ceil for reverse: ceil(q+1.5)=q+2)

        if m % 2 == 0:
            # m even
            x[0::4, :] = uu[:q, :]
            x[1::4, :] = yi[m - 1:q - 1:-1, :]
            x[2::4, :] = yi[:q, :]
            x[3::4, :] = uu[m - 1:q - 1:-1, :]
        else:
            # m odd: q = (m-1)//2
            x[0::4, :] = uu[:q, :]            # 1:q in MATLAB (q elements)
            x[1::4, :] = yi[m - 1:q:-1, :]    # m:-1:q+1 in MATLAB
            x[2::4, :] = yi[:q, :]            # 1:q-1 -> but wait, need to check count
            # Actually let me count more carefully for odd m
            # n = 2*m when m==p, so n is even
            # x has n elements, x[0::4] has n/4 elements = m/2 elements
            # For m odd, that's (m-1)/2 = q... but len(x[0::4]) could be ceil(n/4)
            # n = 2m, so x[0::4] has ceil(2m/4) = ceil(m/2) elements
            # For m odd, ceil(m/2) = (m+1)/2 = q+1
            x[0::4, :] = uu[:q + 1, :]          # q+1 elements
            x[1::4, :] = yi[m - 1:q:-1, :]      # m-1 down to q+1 = m-1-q elements
            x[2::4, :] = yi[:q, :]              # q elements
            x[3::4, :] = uu[m - 1:q:-1, :]      # m-1 down to q+1 = m-1-q elements
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
