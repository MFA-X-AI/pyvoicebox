"""V_HUFFMAN - Calculate a D-ary Huffman code."""

from __future__ import annotations
import numpy as np


def v_huffman(p, a='01') -> tuple[list, np.ndarray, float]:
    """Calculate a D-ary Huffman code.

    Parameters
    ----------
    p : array_like
        Vector of symbol probabilities.
    a : str or list, optional
        Alphabet characters. Length determines code order. Default '01' (binary).

    Returns
    -------
    cc : list
        Code for each symbol (list of strings or lists).
    ll : ndarray
        Code lengths for each symbol.
    l : float
        Average code length.
    """
    p = np.asarray(p, dtype=float).ravel()
    np_ = len(p)
    d = len(a)

    # Append zeros to ensure full code tree
    nx = np_ + (1 - np_) % (d - 1)
    if nx < np_:
        nx += d - 1
    px = np.zeros(nx)
    px[:np_] = p

    cl = (nx - 1) // (d - 1)  # max potential code length
    cd = np.zeros((nx, cl), dtype=int)
    qx = px.copy()
    ix = np.arange(nx)
    kx = np.zeros(nx, dtype=int)

    for i in range(cl - 1, -1, -1):
        nc = 1 + (i + 1) * (d - 1)  # adjust for 0-based
        # Only use first nc elements
        jx = np.argsort(qx[:nc])
        kx_temp = np.zeros(nc, dtype=int)
        kx_temp[jx] = np.arange(nc)

        # Map current indices through kx
        for idx in range(nx):
            if ix[idx] < nc:
                cd[idx, i] = kx_temp[ix[idx]]

        # Update indices
        for idx in range(nx):
            if ix[idx] < nc:
                ix[idx] = max(kx_temp[ix[idx]] - d + 1, 0)

        # Combine d smallest probabilities
        sorted_q = qx[:nc].copy()
        sorted_q.sort()
        combined = np.sum(sorted_q[:d])
        new_q = np.append(sorted_q[d:], combined)
        new_q.sort()
        qx = np.zeros_like(qx)
        qx[:len(new_q)] = new_q

    cc = []
    ll = np.zeros(np_, dtype=int)
    for i in range(np_):
        ci = cd[i, cd[i, :] < d]
        ll[i] = len(ci)
        code = ''.join(a[c] for c in ci)
        cc.append(code)

    total_p = np.sum(p)
    if total_p > 0:
        l = np.dot(p, ll) / total_p
    else:
        l = 0.0

    return cc, ll, l
