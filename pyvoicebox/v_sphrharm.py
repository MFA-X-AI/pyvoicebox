"""V_SPHRHARM - Forward and inverse spherical harmonic transform (stub).

The full implementation is very complex. A basic stub is provided.
"""

from __future__ import annotations
import numpy as np


def v_sphrharm(*args, **kwargs) -> None:
    """Forward and inverse spherical harmonic transform.

    This is a complex function with many modes. A basic version is
    available through scipy.special.sph_harm for individual harmonics.

    Raises
    ------
    NotImplementedError
        Full spherical harmonic transform implementation pending.
        Use scipy.special.sph_harm for individual harmonics.
    """
    raise NotImplementedError(
        "v_sphrharm full implementation pending. "
        "Use scipy.special.sph_harm(m, n, theta, phi) for individual harmonics."
    )
