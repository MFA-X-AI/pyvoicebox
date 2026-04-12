"""V_ROOTSTAB - Determine number of polynomial roots outside, inside and on the unit circle."""

from __future__ import annotations
import numpy as np


def v_rootstab(p) -> tuple[int, int, int]:
    """Determine number of polynomial roots outside, inside and on the unit circle.

    Parameters
    ----------
    p : array_like
        Polynomial with real or complex coefficients.

    Returns
    -------
    no : int
        Number of roots outside the unit circle.
    ni : int
        Number of roots inside the unit circle.
    nc : int
        Number of roots lying on the unit circle.
    """
    p = np.asarray(p, dtype=complex).ravel()

    no = 0
    nc = 0

    if np.all(p == 0):
        ni = 0
        return no, ni, nc

    # Trim leading zeros
    first_nonzero = np.argmax(p != 0)
    p = p[first_nonzero:]

    np0 = len(p)
    npd = 0
    nod = 0

    while len(p) > 1:
        n_p = len(p)
        # Normalize p
        norm = np.sqrt(np.real(np.dot(p, np.conj(p))))
        p = p / norm

        pf = np.conj(p[::-1])
        k = -p[-1] / pf[-1]
        q = p + k * pf
        q[-1] = 0.0  # force exact zero

        if np.all(q == 0):
            # Take derivative
            p = p[:-1] * np.arange(n_p - 1, 0, -1)
            if npd == 0:
                npd = n_p
                nod = no
        elif q[0] == 0:
            # |k|=1 and q != 0
            last_nonzero = np.max(np.where(q != 0)[0])
            q = q[:last_nonzero + 1]
            dr = -q[-1] / (pf[-1] * k)
            if abs(np.real(dr)) > abs(np.imag(dr)):
                cf = abs(np.real(dr))
            else:
                cf = 0.25 * abs(np.imag(dr))
            c = np.sqrt(1 + cf ** 2) - cf
            pad = np.zeros(n_p - len(q))
            q_padded = np.concatenate([pad, q])
            p = p / c + k * c * pf + q_padded
        elif abs(k) > 1:
            last_nonzero = np.max(np.where(q != 0)[0])
            q = q[:last_nonzero + 1]
            p = np.conj(q[::-1])
            no += n_p - len(p)
        else:
            last_nonzero = np.max(np.where(q != 0)[0])
            p = q[:last_nonzero + 1]

    if npd > 0:
        nc = npd - 1 - 2 * (no - nod)
    ni = np0 - 1 - nc - no

    return no, ni, nc
