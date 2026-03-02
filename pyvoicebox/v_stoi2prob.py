"""V_STOI2PROB - Convert STOI to probability."""

import numpy as np


def v_stoi2prob(s, m='i'):
    """Convert STOI to probability.

    Parameters
    ----------
    s : array_like
        Matrix containing STOI values.
    m : str, optional
        Mapping: 'i' for IEEE sentences (default), 'd' for Dantale corpus.

    Returns
    -------
    p : ndarray
        Corresponding probability values.

    References
    ----------
    [1] C. H. Taal et al. An algorithm for intelligibility prediction of
        time-frequency weighted noisy speech. IEEE Trans. Audio, Speech,
        Language Processing, 19(7):2125-2136, 2011.
    """
    s = np.asarray(s, dtype=float)
    if m and m[0] == 'd':
        a = -14.5435
        b = 7.0792
    else:
        a = -17.4906
        b = 9.6921
    p = 1.0 / (1.0 + np.exp(a * s + b))
    return p
