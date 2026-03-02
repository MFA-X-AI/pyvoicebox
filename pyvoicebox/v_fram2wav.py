"""V_FRAM2WAV - Convert frame values to a continuous waveform."""

import numpy as np


def v_fram2wav(x, tt, mode='l'):
    """Convert frame values to a continuous waveform.

    Parameters
    ----------
    x : ndarray, shape (nf,) or (nf, p)
        Input signal: one row per frame.
    tt : ndarray, shape (nf, 2) or (nf, 3)
        Frame specifications. Each row: [start_sample, end_sample, flag].
        flag=1 for start of new spurt. If tt has only 2 columns, spurts
        are auto-detected from gaps.
    mode : str, optional
        'z' for zero-order hold, 'l' for linear interpolation (default).

    Returns
    -------
    w : ndarray, shape (n, p)
        Interpolated waveform of length n = tt[-1, 1].
    s : ndarray, shape (ns, 2)
        Start and end sample numbers of each spurt.
    """
    x = np.asarray(x, dtype=float)
    tt = np.asarray(tt, dtype=float)

    if x.ndim == 1:
        x = x[:, np.newaxis]

    nf, p = x.shape
    n = int(np.round(tt[-1, 1]))
    w = np.full((n, p), np.nan)
    nt = tt.shape[1]

    # MATLAB 1-based indexing: ceil(tt(:,1)) and floor(tt(:,2))
    ix1 = np.ceil(tt[:, 0]).astype(int)   # start of frame sample (1-based)
    ix2 = np.floor(tt[:, 1]).astype(int)  # end of frame sample (1-based)

    # Determine spurt boundaries
    if nt > 2:
        ty = (tt[:, 2] > 0).astype(float)
    else:
        ty = np.zeros(nf)
        ty[1:] = (ix1[1:] > ix2[:-1] + 1).astype(float)

    ty[0] = 1  # first frame always starts a spurt
    # NaN always ends previous spurt
    nan_mask = np.any(np.isnan(x), axis=1)
    ty[nan_mask] = 1
    # NaN always forces a new spurt for the frame after
    nan_idx = np.where(nan_mask[:nf - 1])[0]
    ty[nan_idx + 1] = 1

    ty = ty.astype(float)
    # Encode: bit 0 = starts a spurt, bit 1 = ends a spurt
    # MATLAB: ty(1:end-1) = ty(1:end-1) + 2*ty(2:end)
    ty[:nf - 1] = ty[:nf - 1] + 2 * ty[1:]
    ty[nf - 1] = ty[nf - 1] + 2  # last frame always ends a spurt

    nx = ix2 - ix1 + 1

    if 'z' in mode:
        # Zero-order hold
        for i in range(nf):
            if nx[i] > 0:
                # Convert to 0-based
                w[ix1[i] - 1:ix2[i], :] = np.tile(x[i, :], (nx[i], 1))
    else:
        # Linear interpolation (default)
        ttm = (tt[:, 0] + tt[:, 1]) / 2.0  # mid point of frame (1-based)
        ixm = np.floor(ttm).astype(int)     # end of first half (1-based)

        for i in range(nf):
            if nx[i] > 0:
                tyi = int(ty[i])
                if tyi == 3:  # isolated frame: zero-order hold
                    w[ix1[i] - 1:ix2[i], :] = np.tile(x[i, :], (nx[i], 1))
                else:
                    nxm = ixm[i] - ix1[i] + 1
                    if nxm > 0:
                        if tyi == 1:  # start of spurt
                            grad = (x[i + 1, :] - x[i, :]) / (ttm[i + 1] - ttm[i])
                        else:
                            grad = (x[i, :] - x[i - 1, :]) / (ttm[i] - ttm[i - 1])
                        samples = np.arange(ix1[i], ixm[i] + 1)  # 1-based
                        w[ix1[i] - 1:ixm[i], :] = (
                            np.tile(x[i, :], (nxm, 1)) +
                            (samples[:, np.newaxis] - ttm[i]) * grad[np.newaxis, :]
                        )
                    if nx[i] > nxm:
                        if tyi == 2:  # end of spurt
                            grad = (x[i, :] - x[i - 1, :]) / (ttm[i] - ttm[i - 1])
                        else:
                            grad = (x[i + 1, :] - x[i, :]) / (ttm[i + 1] - ttm[i])
                        n_remaining = ix2[i] - ixm[i]
                        samples = np.arange(ixm[i] + 1, ix2[i] + 1)  # 1-based
                        w[ixm[i]:ix2[i], :] = (
                            np.tile(x[i, :], (n_remaining, 1)) +
                            (samples[:, np.newaxis] - ttm[i]) * grad[np.newaxis, :]
                        )

    # Sort out spurt positions
    ty_clean = ty.copy()
    ty_clean[nan_mask] = 0

    # Find starts (bit 0 set) and ends (bit 1 set)
    starts = np.where(np.bitwise_and(ty_clean.astype(int), 1) > 0)[0]
    ends = np.where(np.bitwise_and(ty_clean.astype(int), 2) > 0)[0]

    s = np.column_stack([ix1[starts], ix2[ends]])

    if p == 1:
        w = w.ravel()

    return w, s
