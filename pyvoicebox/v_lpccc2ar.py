"""V_LPCCC2AR - Convert complex cepstrum to AR coefficients."""

import numpy as np


def v_lpccc2ar(cc):
    """Convert complex cepstrum to AR coefficients.

    Parameters
    ----------
    cc : array_like, shape (nf, p)
        Complex cepstral coefficients.

    Returns
    -------
    ar : ndarray, shape (nf, p+1)
        Autoregressive coefficients.
    """
    cc = np.atleast_2d(np.asarray(cc, dtype=float))
    nf, p = cc.shape
    rp = -np.arange(1, p + 1, dtype=float)
    cc = cc * rp[np.newaxis, :]

    if p < 2:
        ar = np.column_stack([np.ones((nf, 1)), cc[:, 0:1]])
    else:
        ar = np.zeros((nf, p + 1))
        ar[:, 0] = 1.0
        ar[:, 1] = cc[:, 0]
        ar[:, 2] = (cc[:, 1] + cc[:, 0] ** 2) / 2.0
        for k in range(3, p + 1):
            ar[:, k] = (cc[:, k - 1] + np.sum(cc[:, :k - 1] * ar[:, k - 1:0:-1], axis=1)) / k

    return ar
