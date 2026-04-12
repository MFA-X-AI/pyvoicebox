"""V_READFLAC - Read a .FLAC format sound file.

Uses the soundfile library for FLAC decoding.
"""

from __future__ import annotations
import numpy as np


def v_readflac(filename, mode='p') -> tuple[np.ndarray, int]:
    """Read a .FLAC format sound file.

    Parameters
    ----------
    filename : str
        Path to the FLAC file.
    mode : str, optional
        Scaling mode string. Default is 'p'.
        'p' : Scaled so +-1 equals full scale (default).
        'r' : Raw unscaled data (integer values).
        's' : Auto scale to make data peak = +-1.

    Returns
    -------
    y : ndarray
        Data matrix of shape (samples, channels).
    fs : int
        Sample frequency in Hz.
    """
    import soundfile as sf
    import os

    if not os.path.isfile(filename):
        if os.path.isfile(filename + '.flac'):
            filename = filename + '.flac'
        else:
            raise FileNotFoundError(f"Cannot open {filename} for input")

    if not mode:
        mode = 'p'

    sc = 'p'
    for c in mode:
        if c in 'prsq':
            sc = c
            break

    info = sf.info(filename)
    fs = info.samplerate

    if sc == 'r':
        y, _ = sf.read(filename, dtype='int32')
    else:
        y, _ = sf.read(filename, dtype='float64')

        if sc == 's':
            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak

    if y.ndim == 1:
        y = y[:, np.newaxis]

    return y, fs
