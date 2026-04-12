"""V_SPRINTCPX - Format a complex number for printing."""

from __future__ import annotations
import numpy as np


def v_sprintcpx(z, f='g') -> str:
    """Format a complex number for printing.

    Parameters
    ----------
    z : complex
        Complex number to format.
    f : str, optional
        Format string. May include 'i' or 'j' for sqrt(-1) symbol.
        Default 'g'.

    Returns
    -------
    s : str
        Formatted string.
    """
    if not f:
        f = 'g'

    # Determine sqrt(-1) symbol
    if 'i' in f:
        ij = 'i'
    else:
        ij = 'j'

    # Remove i/j from format
    f = f.replace('i', '').replace('j', '')
    if not f:
        f = 'g'

    a = np.real(z)
    b = np.imag(z)

    fmt = f'{f}'

    if a == 0 and b == 0:
        s = format(0, fmt)
    elif b == 0:
        s = format(a, fmt)
    elif a == 0:
        s = f'{format(b, fmt)}{ij}'
    else:
        if b > 0:
            s = f'{format(a, fmt)}+{format(b, fmt)}{ij}'
        else:
            s = f'{format(a, fmt)}{format(b, fmt)}{ij}'

    return s
