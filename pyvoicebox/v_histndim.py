"""V_HISTNDIM - Generate an n-dimensional histogram."""

import numpy as np


def v_histndim(x, b=None, mode=''):
    """Generate an n-dimensional histogram.

    Parameters
    ----------
    x : array_like
        Input data, shape (m, d).
    b : array_like, optional
        Histogram bin specification, shape (3, d) or (1, d) or (3, 1).
        Row 0: number of bins; Row 1: min of first bin; Row 2: max of last bin.
        Default: 10 bins per dimension.
    mode : str, optional
        'p' to scale as probabilities; 'z' for zero base in 2D plot.

    Returns
    -------
    v : ndarray
        Histogram counts (or probabilities if 'p' in mode).
    t : list of ndarray
        Bin boundary values for each dimension.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape

    if b is None:
        b = np.full((1, d), 10, dtype=float)

    b = np.asarray(b, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if b.shape[1] == 1:
        b = np.tile(b, (1, d))

    if b.shape[0] < 3:
        mi = np.min(x, axis=0)
        ma = np.max(x, axis=0)
        w = (ma - mi) / (b[0, :] - 0.001)
        if b.shape[0] < 3:
            b_full = np.zeros((3, d))
            b_full[0, :] = b[0, :]
            b_full[2, :] = ma + 0.0005 * w
            b_full[1, :] = mi - 0.0005 * w
            b = b_full

    acd = np.where(b[0, :] > 0)[0]
    sv = b[0, acd].astype(int)
    nbt = int(np.prod(sv))
    t = []

    ok = np.ones(n, dtype=bool)
    ix = np.full(n, nbt - int(np.sum(np.cumprod(sv))), dtype=int)
    k = 1
    for idx, j in enumerate(acd):
        nbins = int(b[0, j])
        bw = nbins / (b[2, j] - b[1, j])
        bi = np.ceil((x[:, j] - b[1, j]) * bw).astype(int)
        ok = ok & (bi > 0) & (bi <= nbins)
        ix[ok] = ix[ok] + k * bi[ok]
        k = k * nbins
        t.append(b[1, j] + np.arange(nbins + 1) / bw)

    # Build histogram using sparse-like accumulation
    v = np.zeros(nbt)
    valid_ix = ix[ok]
    # Ensure indices are valid (0-based)
    valid_ix = valid_ix - 1  # convert from 1-based to 0-based
    valid_ix = np.clip(valid_ix, 0, nbt - 1)
    np.add.at(v, valid_ix, 1)

    if len(sv) > 1:
        v = v.reshape(sv, order='F')

    if 'p' in mode:
        v = v / n

    return v, t
