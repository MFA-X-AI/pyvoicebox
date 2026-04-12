"""V_CCWARPF - Warp cepstral coefficients."""

from __future__ import annotations
import numpy as np


def v_ccwarpf(f, n, s='ll') -> np.ndarray:
    """Warp cepstral coefficients for frequency/representation changes.

    Parameters
    ----------
    f : array_like
        [original_fs, new_fs]. If scalar, new_fs=1.
    n : array_like
        [original_n, new_n]. If scalar, new_n=original_n.
    s : str, optional
        Two characters: s[0],s[1] = 'l' for linear, 'm' for mel frequency.
        Uppercase if c0 is included.

    Returns
    -------
    m : ndarray
        Warping matrix.
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = np.atleast_1d(np.asarray(n, dtype=int))

    if len(f) < 2:
        f = np.append(f, 1.0)
    if len(n) < 2:
        n = np.append(n, n[0])

    z = np.array([c < 'a' for c in s], dtype=bool)
    s_lower = s.lower()

    if s_lower == 'll':
        k = np.arange(1, n[1] - int(z[1]) + 1)
        ff = (np.arange(1, n[0] + 1) - int(z[0])) * f[1] / f[0]
        fa = 2 * np.sin(ff * np.pi) * ff / np.pi
        fb = ff ** 2
        ka = 1 - 2 * (k % 2)
        kb = k ** 2

        a = fa[:, np.newaxis] * ka[np.newaxis, :]
        b = fb[:, np.newaxis] - kb[np.newaxis, :]

        # Handle exact integer frequencies
        f0 = np.where(np.floor(ff) == ff)[0]
        if len(f0) > 0:
            for idx in f0:
                a[idx, :] = (ff[idx] == k).astype(float)
                b[idx, :] = 1.0

        m_mat = a / b

        if z[1]:
            col0 = np.ones(n[0])
            col0[1:] = 0.5 * fa[1:] / fb[1:]
            m_mat = np.column_stack([col0, m_mat])

        return m_mat

    raise NotImplementedError(f"Mode '{s}' not implemented, only 'll' supported.")
