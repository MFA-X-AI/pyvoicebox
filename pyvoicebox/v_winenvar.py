"""V_WINENVAR - Read Windows environment variable (stub)."""

import os


def v_winenvar(name):
    """Read a Windows environment variable.

    This is a cross-platform stub that uses os.environ.

    Parameters
    ----------
    name : str
        Environment variable name.

    Returns
    -------
    value : str or None
        Value of the environment variable, or None if not found.
    """
    return os.environ.get(name, None)
