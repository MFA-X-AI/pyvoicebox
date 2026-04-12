"""V_REGEXFILES - Find files matching a regular expression pattern."""

from __future__ import annotations
import os
import re


def v_regexfiles(pattern, directory='.', recursive=False) -> list:
    """Find files matching a regular expression pattern.

    Parameters
    ----------
    pattern : str
        Regular expression pattern to match filenames.
    directory : str, optional
        Directory to search. Default is current directory.
    recursive : bool, optional
        If True, search recursively. Default False.

    Returns
    -------
    files : list of str
        List of matching file paths.
    """
    regex = re.compile(pattern)
    matches = []

    if recursive:
        for root, dirs, filenames in os.walk(directory):
            for fn in filenames:
                if regex.search(fn):
                    matches.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(directory):
            if regex.search(fn) and os.path.isfile(os.path.join(directory, fn)):
                matches.append(os.path.join(directory, fn))

    return sorted(matches)
