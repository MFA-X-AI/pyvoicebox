"""V_FOPENMKD - Open file, creating directories if needed."""

import os


def v_fopenmkd(filename, mode='r', **kwargs):
    """Open a file, creating any missing parent directories.

    Parameters
    ----------
    filename : str
        Path to file.
    mode : str, optional
        File open mode (default 'r').
    **kwargs
        Additional arguments passed to open().

    Returns
    -------
    fid : file object
        Open file handle.
    """
    try:
        return open(filename, mode, **kwargs)
    except FileNotFoundError:
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return open(filename, mode, **kwargs)
