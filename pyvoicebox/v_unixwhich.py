"""V_UNIXWHICH - Search system path for an executable (Python equivalent)."""

import shutil


def v_unixwhich(c, e=None):
    """Search system path for an executable program.

    Uses Python's shutil.which() as a cross-platform equivalent.

    Parameters
    ----------
    c : str
        Name of file to search for.
    e : str, optional
        Extensions to search (ignored; shutil.which handles this).

    Returns
    -------
    f : str or None
        Full pathname of executable, or None if not found.
    """
    return shutil.which(c)
