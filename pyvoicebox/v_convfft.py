"""V_CONVFFT - 1-D convolution or correlation using FFT."""

from __future__ import annotations
import numpy as np


class _ConvFFTPrecomputed:
    """Structure holding precomputed FFT of impulse response."""
    def __init__(self):
        self.d = None
        self.nx = None
        self.ns = None
        self.nv = None
        self.vmin = None
        self.vmax = None
        self.nf = None
        self.fh = None
        self.fmin = None
        self.fmax = None
        self.nz = None
        self.zmin = None
        self.zmax = None


def v_convfft(x, h, d=None, m='', h0=1, x1=1, x2=None) -> np.ndarray:
    """1-D convolution or correlation using FFT.

    Parameters
    ----------
    x : array_like or int
        Input array, or size(x, d) when using precompute mode ('z' in m).
        If h is a _ConvFFTPrecomputed object, x is the input array.
    h : array_like or _ConvFFTPrecomputed
        Impulse response, or precomputed structure from previous call with 'z'.
    d : int, optional
        Axis of x to do convolution along (0-based). Default: first non-singleton.
    m : str, optional
        Mode options: 'x' for real correlation, 'X' for complex correlation,
        'z' for precomputation.
    h0 : int, optional
        Origin sample number in h (1-based). Default: 1.
    x1 : int, optional
        Start of range in x to align with origin of h (1-based). Default: 1.
    x2 : int, optional
        End of range in x to align with origin of h (1-based). Default: size(x, d).

    Returns
    -------
    z : ndarray or _ConvFFTPrecomputed
        Convolution output, or precomputed structure if 'z' in m.
    """
    if isinstance(h, _ConvFFTPrecomputed):
        # h is a precomputed structure
        precomp = h
        d = precomp.d
        nx = precomp.nx
        ns = precomp.ns
        nv = precomp.nv
        vmin = precomp.vmin
        vmax = precomp.vmax
        nf = precomp.nf
        fh = precomp.fh
        fmin = precomp.fmin
        fmax = precomp.fmax
        nz = precomp.nz
        zmin = precomp.zmin
        zmax = precomp.zmax

        x = np.asarray(x, dtype=complex)
        s = list(x.shape)
        k = x.size // nx
        if x.shape[d] != nx or len(s) != ns:
            raise ValueError('input x has incompatible dimensions')

        # Reshape x
        if d == 0:
            v = x.reshape(nx, k)
        else:
            perm = list(range(d, ns)) + list(range(0, d))
            v = np.transpose(x, perm).reshape(nx, k)

        if nv < nx:
            v = v[vmin:vmax + 1, :]  # 0-based inclusive

        # Do the convolution
        zz = np.fft.ifft(np.fft.fft(v, n=nf, axis=0) * fh[:, np.newaxis], axis=0)
        z_out = np.zeros((nz, k), dtype=complex)
        z_out[zmin:zmax + 1, :] = zz[fmin:fmax + 1, :]

        if np.all(np.isreal(x)):
            z_out = np.real(z_out)

        s[d] = nz
        if d == 0:
            z_out = z_out.reshape(s)
        else:
            perm_shape = [s[i] for i in (list(range(d, ns)) + list(range(0, d)))]
            z_out = z_out.reshape(perm_shape)
            inv_perm = list(range(ns - d, ns)) + list(range(0, ns - d))
            z_out = np.transpose(z_out, inv_perm)

        return z_out

    # Normal input calling sequence
    x = np.asarray(x)
    h = np.asarray(h, dtype=complex).ravel()

    s = list(x.shape)
    ps = x.size
    ns = len(s)

    if 'z' in m:
        # Output pre-computed structure
        if d is None:
            raise ValueError('d must be specified explicitly')
        if x.size != 1:
            raise ValueError('x must equal size(*, d)')
        nx = int(x.ravel()[0])
    else:
        if d is None:
            if ps < 2:
                d = 0
            else:
                for i, si in enumerate(s):
                    if si > 1:
                        d = i
                        break
                if d is None:
                    d = 0
        nx = s[d]

    k = ps // nx if ps > 1 else 1

    if x2 is None:
        x2 = nx

    nz = x2 - x1 + 1  # number of output lags
    nh = len(h)

    if 'X' in m:
        h = np.conj(h[::-1])
        h0 = nh + 1 - h0
    elif 'x' in m:
        h = h[::-1].copy()
        h0 = nh + 1 - h0

    hmin = h0 + x1 - nx
    hmax = h0 + x2 - 1
    xmin = x1 + h0 - nh
    xmax = x2 + h0 - 1

    if hmin > 1 or hmax < nh:
        hmin_c = max(hmin, 1)
        hmax_c = min(hmax, nh)
        h = h[hmin_c - 1:hmax_c]  # 1-based to 0-based
        nh = len(h)
        h0 = h0 - hmin_c + 1

    if xmin > 1 or xmax < nx:
        vmin = max(xmin, 1)
        vmax = min(xmax, nx)
        x1_new = x1 - vmin + h0
        x2_new = x2 - vmin + h0
    else:
        vmin = 1
        vmax = nx
        x1_new = x1 + h0 - 1
        x2_new = x2 + h0 - 1

    nv = vmax - vmin + 1
    nxz = min(max(max(nh - x1_new, 0), max(x2_new - nv, 0)), nh - 1)

    # Round up to next power of 2
    nf = int(2 ** np.ceil(np.log2(nv + nxz)))

    fmin_idx = max(x1_new, 1)
    fmax_idx = min(x2_new, min(nf, nx + nh - 1))
    zmin_idx = max(1, 2 - x1_new)
    zmax_idx = zmin_idx + fmax_idx - fmin_idx

    fh = np.fft.fft(h, n=nf)

    if 'z' in m:
        # Save as precomputed structure
        result = _ConvFFTPrecomputed()
        result.d = d
        result.nx = nx
        result.ns = ns
        result.nv = nv
        result.vmin = vmin - 1  # convert to 0-based
        result.vmax = vmax - 1  # convert to 0-based
        result.nf = nf
        result.fh = fh
        result.fmin = fmin_idx - 1  # convert to 0-based
        result.fmax = fmax_idx - 1  # convert to 0-based
        result.nz = nz
        result.zmin = zmin_idx - 1  # convert to 0-based
        result.zmax = zmax_idx - 1  # convert to 0-based
        return result

    if x.size > 0:
        # Reshape x
        if d == 0:
            v = x.reshape(nx, k)
        else:
            perm = list(range(d, ns)) + list(range(0, d))
            v = np.transpose(x, perm).reshape(nx, k)

        if nv < nx:
            v = v[vmin - 1:vmax, :]  # 1-based to 0-based

        v = np.asarray(v, dtype=complex)
        zz = np.fft.ifft(np.fft.fft(v, n=nf, axis=0) * fh[:, np.newaxis], axis=0)
        z_out = np.zeros((nz, k), dtype=complex)
        z_out[zmin_idx - 1:zmax_idx, :] = zz[fmin_idx - 1:fmax_idx, :]

        if np.all(np.isreal(x)):
            z_out = np.real(z_out)

        s[d] = nz
        if d == 0:
            z_out = z_out.reshape(s)
        else:
            perm_shape = [s[i] for i in (list(range(d, ns)) + list(range(0, d)))]
            z_out = z_out.reshape(perm_shape)
            inv_perm = list(range(ns - d, ns)) + list(range(0, ns - d))
            z_out = np.transpose(z_out, inv_perm)

        return z_out
    else:
        return np.array([])
