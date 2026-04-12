"""V_LPCSS2ZZ - Convert s-plane poles to z-plane poles."""

from __future__ import annotations
import numpy as np


def v_lpcss2zz(ss, nr=None) -> np.ndarray:
    """Convert s-plane poles to z-plane poles.

    Parameters
    ----------
    ss : array_like, shape (n, q)
        S-plane pole positions in normalized-Hz units.
    nr : int, optional
        Number of poles that should NOT be supplemented by conjugate pairs.
        If nr=-1, conjugate of any column containing a non-real number.

    Returns
    -------
    zz : ndarray, shape (n, p)
        Z-plane poles.
    """
    ss = np.atleast_2d(np.asarray(ss, dtype=complex))
    if nr is not None and nr < ss.shape[1]:
        if nr >= 0:
            ss = np.column_stack([ss, np.conj(ss[:, nr:])])
        else:
            # Conjugate of columns containing non-real numbers
            has_imag = np.any(np.imag(ss) != 0, axis=0)
            ss = np.column_stack([ss, np.conj(ss[:, has_imag])])
    zz = np.exp(2 * np.pi * ss)
    return zz
