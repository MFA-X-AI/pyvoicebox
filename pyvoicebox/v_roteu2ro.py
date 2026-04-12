"""V_ROTEU2RO - convert Euler angles to rotation matrix."""

from __future__ import annotations
import numpy as np
from pyvoicebox.v_roteu2qr import v_roteu2qr
from pyvoicebox.v_rotqr2ro import v_rotqr2ro


def v_roteu2ro(m, e=None) -> np.ndarray:
    """Convert Euler angles to rotation matrix.

    Parameters
    ----------
    m : str
        Rotation code string.
    e : array_like, optional
        Euler angles.

    Returns
    -------
    r : ndarray, shape (3, 3) or (3, 3, N)
        Rotation matrix/matrices.
    """
    if e is None:
        return v_rotqr2ro(v_roteu2qr(m))
    else:
        return v_rotqr2ro(v_roteu2qr(m, e))
