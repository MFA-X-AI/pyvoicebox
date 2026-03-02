"""V_ROTQR2EU - convert real quaternion to Euler angles."""

import numpy as np
from pyvoicebox.v_rotro2eu import v_rotro2eu
from pyvoicebox.v_rotqr2ro import v_rotqr2ro


def v_rotqr2eu(m, q):
    """Convert real quaternion to Euler angles.

    Parameters
    ----------
    m : str
        Rotation code string.
    q : array_like, shape (4,) or (4, N)
        Quaternion(s) [w, x, y, z].

    Returns
    -------
    e : ndarray, shape (K,) or (K, N)
        Euler angles.
    """
    return v_rotro2eu(m, v_rotqr2ro(q))
