"""V_LPCPF2CC - Convert power spectrum to complex cepstrum."""

from __future__ import annotations
import numpy as np
from pyvoicebox.v_rsfft import v_rsfft


def v_lpcpf2cc(pf, np_out=None, f=None) -> tuple[np.ndarray, np.ndarray]:
    """Convert power spectrum to complex cepstrum.

    Parameters
    ----------
    pf : array_like, shape (nf, n)
        Power spectrum, uniformly spaced DC to Nyquist.
    np_out : int, optional
        Number of cepstral coefficients to calculate. Default is n-1.
    f : array_like, shape (n,), optional
        Frequencies of pf columns.

    Returns
    -------
    cc : ndarray, shape (nf, np_out)
        Complex cepstral coefficients.
    c0 : ndarray, shape (nf, 1)
        Zeroth cepstral coefficient.
    """
    pf = np.atleast_2d(np.asarray(pf, dtype=float))
    nf, nq = pf.shape
    if np_out is None:
        np_out = nq - 1

    if f is None:
        cc = v_rsfft(np.log(pf).T).T / (2 * nq - 2)
        c0 = cc[:, 0:1] * 0.5
        cc[:, nq - 1] = cc[:, nq - 1] * 0.5
        if np_out > nq - 1:
            cc = np.column_stack([cc[:, 1:nq], np.zeros((nf, np_out - nq + 1))])
        else:
            cc = cc[:, 1:np_out + 1]
    else:
        f = np.asarray(f, dtype=float)
        nm = min(np_out, nq - 1)
        cos_matrix = np.cos(2 * np.pi * f[:, np.newaxis] * np.arange(nm + 1)[np.newaxis, :])
        cc = 0.5 * np.linalg.lstsq(cos_matrix, np.log(pf).T, rcond=None)[0].T
        c0 = cc[:, 0:1]
        cc = cc[:, 1:]
        if np_out > nq - 1:
            cc = np.column_stack([cc, np.zeros((nf, np_out - nq + 1))])

    return cc, c0
