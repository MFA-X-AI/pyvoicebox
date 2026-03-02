"""V_LPCPF2FF - Convert power spectrum to complex spectrum."""

import numpy as np
from pyvoicebox.v_lpcpf2cc import v_lpcpf2cc
from pyvoicebox.v_lpccc2ff import v_lpccc2ff


def v_lpcpf2ff(pf, np_out=None, fi=None):
    """Convert power spectrum to complex spectrum.

    Parameters
    ----------
    pf : array_like, shape (nf, n)
        Power spectrum.
    np_out : int, optional
        Number of complex cepstral coefficients to use.
    fi : array_like, optional
        Vector of frequencies.

    Returns
    -------
    ff : ndarray, shape (nf, n)
        Complex spectrum.
    fo : ndarray
        Vector of frequencies.
    """
    pf = np.atleast_2d(np.asarray(pf, dtype=float))
    nf, nq = pf.shape
    if fi is None:
        if np_out is None:
            np_out = nq - 1
    else:
        if np_out is None:
            np_out = nq - 1

    cc, c0 = v_lpcpf2cc(pf, np_out, fi)
    if fi is None:
        fi_val = nq - 1
    else:
        fi_val = fi
    fx, fo = v_lpccc2ff(cc, fi_val, nc=-1, c0=c0)
    ff = np.sqrt(pf) * np.exp(1j * np.angle(fx))

    return ff, fo
