"""V_LPCAUTO - Perform autocorrelation LPC analysis."""

import numpy as np
from pyvoicebox.v_windows import v_windows


def v_lpcauto(s, p=12, t=None, w='m', m=''):
    """Perform autocorrelation LPC analysis.

    Parameters
    ----------
    s : array_like, shape (ns,) or (ns, nch)
        Input signal.
    p : int, optional
        LPC order. Default is 12.
    t : array_like, optional
        Frame size details [hop, anal, skip]. Default is entire signal.
    w : str or array_like, optional
        Window type or window vector. Default is 'm' (Hamming).
    m : str, optional
        Mode options: 'e' normalize window, 'j' joint analysis.

    Returns
    -------
    ar : ndarray, shape (nf, p+1) or (nf, p+1, nch)
        AR coefficients with ar[:, 0] = 1.
    e : ndarray, shape (nf,) or (nf, nch)
        Residual energy.
    k : ndarray, shape (nf, 2)
        First and last sample of analysis intervals (1-based).
    """
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s[:, np.newaxis]
    elif s.shape[0] == 1:
        s = s.T

    ns, nch = s.shape

    wnam = ['c', 'k', 'm', 'n', 'r', 'w', 's', 'v']
    wtypes = [10, 11, 3, 2, 1, 8, 3, 7]

    modee = 'e' in m
    modej = 'j' in m
    wch = isinstance(w, str)

    if wch:
        if modee:
            wopt = 'e'
        else:
            wopt = ''
        try:
            wch_idx = wnam.index(w)
        except ValueError:
            wch_idx = 2  # default to Hamming
        wch = wch_idx
    else:
        w = np.asarray(w, dtype=float).ravel()
        if modee:
            w = w / np.sqrt(np.dot(w, w))
        wch = False

    if t is None:
        t = np.array([[ns, ns, 0]])
    else:
        t = np.atleast_2d(np.asarray(t, dtype=float))

    nf_t, ng = t.shape
    if ng == 1:
        t = np.column_stack([t, t, np.zeros((nf_t, 1))])
    elif ng == 2:
        t = np.column_stack([t, np.zeros((nf_t, 1))])

    if nf_t == 1:
        nf = int(np.floor(1 + (ns - t[0, 1] - t[0, 2]) / t[0, 0]))
        t = np.tile(t, (nf, 1))
    else:
        nf = nf_t

    # Calculate analysis frame start/end (1-based)
    cumhop = np.cumsum(np.concatenate([[0], t[:nf-1, 0]])) + t[:, 2]
    k_out = np.round(np.column_stack([cumhop + 1, cumhop + t[:, 1]])).astype(int)
    kd = k_out[:, 1] - k_out[:, 0] + 1  # frame lengths
    ku = np.unique(kd)
    nk = len(ku)

    ar = np.zeros((nf, p + 1, nch) if not modej else (nf, p + 1))
    e = np.zeros((nf, nch) if not modej else (nf,))

    for ik in range(nk):
        kui = ku[ik]
        if wch is not False:
            win = v_windows(wtypes[wch], kui, wopt, 5).ravel()
        else:
            win = w[:kui]

        pk = min(p, kui)
        km = kd == kui
        nkm = np.sum(km)
        frame_indices = np.where(km)[0]

        if modej:
            ark = np.zeros((pk + 1, nkm))
            rr = np.zeros((pk + 1, nkm))

            for idx, fi in enumerate(frame_indices):
                start = k_out[fi, 0] - 1
                end = k_out[fi, 1]
                frame_data = s[start:end, :] * win[:end - start, np.newaxis]
                rr[0, idx] = np.sum(frame_data ** 2)
                for lag in range(1, pk + 1):
                    rr[lag, idx] = np.sum(frame_data[:kui - lag, :] * frame_data[lag:, :])

            ark[0, :] = 1.0
            ark[1, :] = -rr[1, :] / rr[0, :]
            ek = rr[0, :] * (ark[1, :] ** 2 - 1)
            for n in range(2, pk + 1):
                ka = (rr[n, :] + np.sum(rr[n-1:0:-1, :] * ark[1:n, :], axis=0)) / ek
                ark[1:n, :] = ark[1:n, :] + ka[np.newaxis, :] * ark[n-1:0:-1, :]
                ark[n, :] = ka
                ek = ek * (1 - ka ** 2)

            ar[frame_indices, :pk + 1] = ark.T
            e[frame_indices] = -ek
        else:
            for ch in range(nch):
                ark = np.zeros((pk + 1, nkm))
                rr = np.zeros((pk + 1, nkm))

                for idx, fi in enumerate(frame_indices):
                    start = k_out[fi, 0] - 1
                    end = k_out[fi, 1]
                    frame_data = s[start:end, ch] * win[:end - start]
                    rr[0, idx] = np.sum(frame_data ** 2)
                    for lag in range(1, pk + 1):
                        rr[lag, idx] = np.sum(frame_data[:kui - lag] * frame_data[lag:])

                ark[0, :] = 1.0
                ark[1, :] = -rr[1, :] / rr[0, :]
                ek = rr[0, :] * (ark[1, :] ** 2 - 1)
                for n in range(2, pk + 1):
                    ka = (rr[n, :] + np.sum(rr[n-1:0:-1, :] * ark[1:n, :], axis=0)) / ek
                    ark[1:n, :] = ark[1:n, :] + ka[np.newaxis, :] * ark[n-1:0:-1, :]
                    ark[n, :] = ka
                    ek = ek * (1 - ka ** 2)

                ar[frame_indices, :pk + 1, ch] = ark.T
                e[frame_indices, ch] = -ek

    # Squeeze single channel
    if nch == 1 and not modej:
        ar = ar[:, :, 0]
        e = e[:, 0]

    return ar, e, k_out
