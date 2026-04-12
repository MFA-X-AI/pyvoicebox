"""V_LPCCONV - Convert between LPC parameter sets (generates conversion string)."""

from __future__ import annotations

def v_lpcconv(from_type, to_type) -> None:
    """Generate conversion path between LPC parameter sets.

    Parameters
    ----------
    from_type : str
        Source parameter type (2-char code).
    to_type : str
        Target parameter type (2-char code).

    Returns
    -------
    path : str
        String describing the conversion path.
    """
    # This is a routing/dispatch function in MATLAB that generates eval strings.
    # In Python, direct function calls are preferred.
    # This is provided for API compatibility but is not typically used.
    raise NotImplementedError(
        "v_lpcconv generates MATLAB eval strings. "
        "In Python, call conversion functions directly, e.g., v_lpcar2rf(ar)."
    )
