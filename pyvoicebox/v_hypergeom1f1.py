"""V_HYPERGEOM1F1 - Confluent hypergeometric function 1F1 (Kummer's M)."""

import numpy as np
from pyvoicebox.v_gammalns import v_gammalns


def v_hypergeom1f1(a, b, z, tol=1e-10, maxj=500, th=30):
    """Confluent hypergeometric function 1F1(a; b; z) = M(a; b; z).

    Parameters
    ----------
    a : float
        First parameter (real scalar).
    b : float
        Second parameter (real scalar).
    z : array_like
        Input values (real).
    tol : float, optional
        Tolerance. Default 1e-10.
    maxj : int, optional
        Maximum iterations. Default 500.
    th : float, optional
        Threshold for algorithm selection. Default 30.

    Returns
    -------
    h : ndarray
        Output values.
    l : ndarray
        Number of iterations used.
    """
    z = np.asarray(z, dtype=float)
    scalar_input = z.ndim == 0
    z = np.atleast_1d(z)
    h = np.zeros_like(z, dtype=float)
    l = np.zeros_like(z, dtype=int)

    a1 = a - 1
    b1 = b - 1
    ba = b - a

    for idx in np.ndindex(z.shape):
        y = z[idx]
        q = False  # break criterion

        if abs(y) < th:
            # Algorithm 1: series expansion (13.2.2)
            d = 1.0
            g = 1.0
            jlim = 0
            k = (b1 + y) ** 2 - 4 * a1 * y
            if k >= 0:
                jlim = max(jlim, 0.5 * (np.sqrt(k) - (b1 + y)))
            k = (b1 - y) ** 2 + 4 * a1 * y
            if k >= 0:
                jlim = max(jlim, 0.5 * (np.sqrt(k) - (b1 - y)))
            jlim = min(maxj, jlim)

            for j in range(1, maxj + 1):
                d = d * y * (a1 + j) / (j * (b1 + j))
                g = g + d
                p = abs(d) < tol * abs(g)
                if q and p and j >= jlim:
                    break
                q = p

        elif y > 0:
            # Algorithm 2: asymptotic for large positive y
            d = 1.0
            g = 1.0
            jlim = 1
            k = (ba - 1 - a + y) ** 2 + 4 * a * (ba - 1)
            if k >= 0:
                jlim = max(jlim, 0.5 * (np.sqrt(k) - (ba - a - 1 + y)))
            k = (ba - a - 1 - y) ** 2 + 4 * a * (ba - 1)
            if k >= 0:
                jlim = max(jlim, 0.5 * (np.sqrt(k) - (ba - a - 1 - y)))
            jlim = int(min(maxj, jlim))

            for j in range(1, jlim + 1):
                d = d * (ba - 1 + j) * (j - a) / (j * y)
                g = g + d
                p = abs(d) < tol * abs(g)
                if q and p:
                    break
                q = p

            gl, gs = v_gammalns(np.array([a, b]))
            g = gs[0] * gs[1] * np.exp(y + gl[1] - gl[0] + (a - b) * np.log(y)) * g

        else:
            # Algorithm 3: large negative y using M(a;b;z) = exp(z)*M(b-a;b;-z)
            d = 1.0
            g = 1.0
            jlim = 1
            k = (a1 - ba + y) ** 2 + 4 * a1 * ba
            if k >= 0:
                jlim = max(jlim, 0.5 * (np.sqrt(k) - (a1 - ba + y)))
            k = (a1 - ba - y) ** 2 + 4 * a1 * ba
            if k >= 0:
                jlim = max(jlim, 0.5 * (np.sqrt(k) - (a1 - ba - y)))
            jlim = int(min(maxj, jlim))

            for j in range(1, jlim + 1):
                d = d * (a1 + j) * (ba - j) / (j * y)
                g = g + d
                p = abs(d) < tol * abs(g)
                if q and p:
                    break
                q = p

            gl, gs = v_gammalns(np.array([ba, b]))
            g = gs[0] * gs[1] * np.exp(gl[1] - gl[0] - a * np.log(-y)) * g

        h[idx] = g
        l[idx] = j

    if scalar_input:
        return h.item(), l.item()
    return h, l
