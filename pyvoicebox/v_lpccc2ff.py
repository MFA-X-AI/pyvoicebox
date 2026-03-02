"""V_LPCCC2FF - Convert complex cepstrum to complex spectrum."""

import numpy as np
from pyvoicebox.v_rfft import v_rfft
from pyvoicebox.v_lpccc2cc import v_lpccc2cc


def v_lpccc2ff(cc, np_out=None, nc=None, c0=None):
    """Convert complex cepstrum to complex spectrum.

    Parameters
    ----------
    cc : array_like, shape (nf, n)
        Complex cepstral coefficients excluding c(0).
    np_out : int, optional
        Size of output spectrum is np_out+1. Default is n.
    nc : int, optional
        Number of cepstral coefficients to use. Set nc=-1 to use n.
    c0 : array_like, shape (nf, 1), optional
        Cepstral coefficient c(0). Default is 0.

    Returns
    -------
    ff : ndarray, shape (nf, np_out+1)
        Complex spectrum from DC to Nyquist.
    f : ndarray
        Normalized frequencies (0 to 0.5).
    """
    cc = np.atleast_2d(np.asarray(cc, dtype=float))
    nf, mc = cc.shape
    if np_out is None:
        np_out = mc
    if nc is not None and nc == -1:
        nc = mc
    if c0 is None:
        c0 = np.zeros((nf, 1))
    else:
        c0 = np.asarray(c0, dtype=float).reshape(nf, 1)

    if nc is None:
        nc = np_out
    if nc == mc:
        combined = np.column_stack([c0, cc])
        ff = np.exp(v_rfft(combined.T, 2 * np_out).T)
    else:
        combined = np.column_stack([c0, v_lpccc2cc(cc, nc)])
        ff = np.exp(v_rfft(combined.T, 2 * np_out).T)

    f = np.linspace(0, 0.5, np_out + 1)
    return ff, f
