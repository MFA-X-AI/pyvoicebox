"""V_PDFMOMENTS - Convert between central moments, raw moments and cumulants."""

from __future__ import annotations
import numpy as np
from scipy.special import comb


def v_pdfmoments(t, m, b=0, a=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert between central moments, raw moments and cumulants.

    Parameters
    ----------
    t : str
        Input/output type string. Lower case = input type, upper case = output type.
        'm'/'M' = central moments, 'r'/'R' = raw moments, 'k'/'K' = cumulants.
    m : array_like
        Vector of input moments; m[0] is always the mean.
    b : float, optional
        Output moments are for a*x + b (default 0).
    a : float, optional
        Output moments are for a*x + b (default 1).

    Returns
    -------
    c : ndarray
        Central moments (or as determined by 'R' or 'K' options).
    r : ndarray
        Raw moments.
    k : ndarray
        Cumulants.
    """
    m = np.asarray(m, dtype=float).ravel()
    n = len(m)

    # Precompute binomial coefficients
    bc = np.zeros((n, n + 1))
    for i in range(n):
        for j in range(i + 2):
            bc[i, j] = comb(i + 1, j, exact=True)

    # Precompute cumulant coefficients
    # This is a complex recursive algorithm; we implement it following the MATLAB code
    fa = np.ones(max(1, n // 2))
    for i in range(1, len(fa)):
        fa[i] = (i + 1) * fa[i - 1]

    mk = [None] * n
    # Initialize: mk[0] = [[1, 1, 1]] meaning powers=1, coefs k->m=1, m->k=1
    mk[0] = np.array([[1, 1, 1]])
    if n > 1:
        mk[1] = np.array([[0, 1, 1, 1]])  # for moment 3 (index 2)

    mn = np.zeros((n, n))
    mn[0, 0] = 1  # mn[0,0] = 1
    if n > 1:
        mn[1, 0] = 1
        mn[1, 1] = 1

    for i in range(3, n):
        j = (i + 1) // 2  # first coefficient row to sum
        nr = 1
        for rr in range(j - 1, i - 2):
            col_idx = i - rr - 3
            if rr < mn.shape[0] and col_idx >= 0 and col_idx < mn.shape[1]:
                nr += int(mn[rr, col_idx])

        mki = np.zeros((nr, i + 2))
        ix = 0
        mki[0, i - 2] = 1  # power of moment i-1
        mki[0, i] = 1  # coef k->m
        mki[0, i + 1] = 1  # coef m->k

        ix = 1
        for rr in range(j, i - 1):
            nk_idx = rr - 1
            col_idx = i - rr - 3
            if nk_idx >= len(mk) or mk[nk_idx] is None:
                continue
            if nk_idx < mn.shape[0] and col_idx >= 0 and col_idx < mn.shape[1]:
                nk = int(mn[nk_idx, col_idx])
            else:
                nk = 0
            if nk == 0:
                continue

            mkk = mk[nk_idx]
            mkik = mkk[:nk, :rr].copy()
            col_to_inc = i - rr - 2
            if col_to_inc < mkik.shape[1]:
                mkik[:, col_to_inc] += 1
            if ix + nk <= mki.shape[0]:
                mki[ix:ix + nk, :rr] = mkik
                # Calculate coefficient for k->m
                for qq in range(nk):
                    if col_to_inc < mkik.shape[1]:
                        mki[ix + qq, i] = mkk[qq, rr] * bc[i - 1, i - rr] / mkik[qq, col_to_inc]
                    rho = np.sum(mkik[qq, :]) - 1
                    rho_int = int(rho)
                    if rho_int > 0 and rho_int <= len(fa):
                        mki[ix + qq, i + 1] = mki[ix + qq, i] * fa[rho_int - 1] * (-1) ** rho_int
                    elif rho_int == 0:
                        mki[ix + qq, i + 1] = mki[ix + qq, i]
                ix += nk

        # Sort rows
        if ix > 0:
            mki = mki[:ix]
            idx = np.lexsort(mki[:, :i].T[::-1])
            mki = mki[idx]

        mk[i - 1] = mki
        # Update mn
        if i - 1 < mn.shape[0]:
            mn[i - 1, 0] = mki.shape[0]
            for cc in range(1, min(i - 1, mn.shape[1])):
                mn[i - 1, cc] = np.sum(np.all(mki[:, :cc] == 0, axis=1))

    # Apply scaling
    mu = a * m[0] + b
    c = m.copy()
    r = m.copy()
    k_out = m.copy()
    m_row = m.copy()

    # Determine input type
    if 'k' in t:
        tin = 3
        k_out = k_out * (a ** np.arange(1, n + 1))
        k_out[0] = 0
    elif 'r' in t:
        tin = 2
    else:
        tin = 1
        c = c * (a ** np.arange(1, n + 1))
        c[0] = 0

    tout = [
        'K' not in t and 'R' not in t,  # output c (central moments)
        True,  # always compute r
        True,  # always compute k
    ]

    for il in range(2):
        # Convert between moments
        if il == 0:
            v = np.concatenate(([1], m_row * (a ** np.arange(1, n + 1))))
            bb = b - mu
            doit = tin == 2 and (tout[0] or tout[2])
        else:
            if tin == 2:
                bb = b
            else:
                v = np.concatenate(([1], c))
                bb = mu
            doit = tout[1]

        if doit:
            y = v[1:].copy()
            if bb != 0:
                for i in range(n):
                    # polyval: bc[i, 0:i+2] * v[0:i+2] evaluated at bb
                    coeffs = bc[i, :i + 2] * v[:i + 2]
                    y[i] = np.polyval(coeffs[::-1], bb)
            if il == 0:
                c = y.copy()
            else:
                r = y.copy()

        # Convert cumulants to/from moments
        if il == 0:
            x = k_out.copy()
            doit = tin == 3 and (tout[0] or tout[1])
        else:
            x = c.copy()
            doit = tin < 3 and tout[2]

        if doit:
            y = x.copy()
            for i in range(3, n):
                if i - 1 < len(mk) and mk[i - 1] is not None:
                    mki = mk[i - 1]
                    col_idx = i + il  # il=0 -> column i (k->m), il=1 -> column i+1 (m->k)
                    if col_idx < mki.shape[1]:
                        terms = mki[:, col_idx] * np.prod(
                            np.power(
                                np.tile(x[1:i + 1], (mki.shape[0], 1)) if i + 1 <= len(x) else np.tile(np.concatenate([x[1:], np.zeros(i + 1 - len(x))]), (mki.shape[0], 1)),
                                mki[:, :i]
                            ),
                            axis=1
                        )
                        y[i] = np.sum(terms)
            if il == 0:
                c = y.copy()
            else:
                k_out = y.copy()

    c[0] = mu
    k_out[0] = mu

    if 'R' in t:
        c = r
    elif 'K' in t:
        c = k_out

    return c, r, k_out
