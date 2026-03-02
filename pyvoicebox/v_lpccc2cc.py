"""V_LPCCC2CC - Extrapolate complex cepstrum."""

import numpy as np


def v_lpccc2cc(cc, np_out=None):
    """Extrapolate complex cepstrum.

    Parameters
    ----------
    cc : array_like, shape (nf, p)
        Complex cepstral coefficients.
    np_out : int, optional
        Output number of coefficients. Default is p.

    Returns
    -------
    c : ndarray, shape (nf, np_out)
        Extrapolated complex cepstral coefficients.
    """
    cc = np.atleast_2d(np.asarray(cc, dtype=float))
    p = cc.shape[1]
    if np_out is None:
        np_out = p

    if np_out <= p:
        c = cc[:, :np_out]
    else:
        from pyvoicebox.v_lpcar2cc import v_lpcar2cc
        from pyvoicebox.v_lpccc2ar import v_lpccc2ar
        c, _ = v_lpcar2cc(v_lpccc2ar(cc), np_out)

    return c
