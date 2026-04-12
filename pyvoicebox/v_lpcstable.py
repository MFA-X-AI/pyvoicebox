"""V_LPCSTABLE - Test AR coefficients for stability and stabilize if necessary."""

from __future__ import annotations
import numpy as np


def v_lpcstable(ar) -> tuple[np.ndarray, np.ndarray]:
    """Test AR coefficients for stability and stabilize if necessary.

    Parameters
    ----------
    ar : array_like, shape (nf, p+1)
        Autoregressive coefficients.

    Returns
    -------
    m : ndarray, shape (nf,)
        Boolean mask identifying stable polynomials.
    a : ndarray, shape (nf, p+1)
        Stabilized polynomials with a[:, 0] = 1.
    """
    ar = np.atleast_2d(np.asarray(ar, dtype=float))
    nf, p1 = ar.shape

    # Ensure leading coefficient is 1
    mm = ar[:, 0] != 1
    if np.any(mm):
        ar = ar.copy()
        ar[mm, :] = ar[mm, :] / ar[mm, 0:1]

    if p1 == 1:
        m = np.ones(nf, dtype=bool)
    elif p1 == 2:
        m = np.abs(ar[:, 1]) < 1
    else:
        rf = ar.copy()
        k = rf[:, p1 - 1].copy()
        m = np.abs(k) < 1

        if np.any(m):
            d = (1 - k[m] ** 2) ** (-1)
            rf[np.ix_(m, range(1, p1 - 1))] = (
                (rf[np.ix_(m, range(1, p1 - 1))] -
                 k[m, np.newaxis] * rf[np.ix_(m, range(p1 - 2, 0, -1))]) *
                d[:, np.newaxis]
            )
            for j in range(p1 - 2, 1, -1):
                k_m = rf[m, j]
                still_stable = np.abs(k_m) < 1
                # Update m: only those that were True and remain stable
                m_indices = np.where(m)[0]
                m[m_indices[~still_stable]] = False
                if not np.any(m):
                    break
                d = (1 - k_m[still_stable] ** 2) ** (-1)
                m_indices = np.where(m)[0]
                rf[np.ix_(m_indices, range(1, j))] = (
                    (rf[np.ix_(m_indices, range(1, j))] -
                     k_m[still_stable, np.newaxis] * rf[np.ix_(m_indices, range(j - 1, 0, -1))]) *
                    d[:, np.newaxis]
                )

    # Stabilize unstable polynomials
    a = ar.copy()
    if not np.all(m):
        for i in np.where(~m)[0]:
            z = np.roots(a[i, :])
            k_mask = np.abs(z) > 1
            z[k_mask] = np.conj(z[k_mask]) ** (-1)
            a[i, :] = np.real(np.poly(z))

    return m, a
