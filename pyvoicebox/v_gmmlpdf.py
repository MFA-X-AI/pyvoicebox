"""V_GMMLPDF - Obsolete wrapper for v_gaussmixp."""

from __future__ import annotations
from .v_gaussmixp import v_gaussmixp


def v_gmmlpdf(*args, **kwargs) -> np.ndarray:
    """Obsolete function - please use v_gaussmixp instead."""
    return v_gaussmixp(*args, **kwargs)
