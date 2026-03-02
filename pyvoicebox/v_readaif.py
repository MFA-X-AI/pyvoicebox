"""V_READAIF - Read a .AIF (AIFF) format sound file.

Uses the soundfile library when available for robust AIFF reading.
"""

import numpy as np


def v_readaif(filename, mode='p', nmax=-1, nskip=0):
    """Read a .AIF (AIFF) format sound file.

    Parameters
    ----------
    filename : str
        Path to the AIF file (with or without .aif extension).
    mode : str, optional
        Scaling mode string. Default is 'p'.
        'p' : Scaled so +-1 equals full scale (default).
        'r' : Raw unscaled data (integer values).
        's' : Auto scale to make data peak = +-1.
    nmax : int, optional
        Maximum number of samples to read. -1 for unlimited (default).
    nskip : int, optional
        Number of samples to skip from start. Default is 0.

    Returns
    -------
    y : ndarray
        Data matrix of shape (samples, channels).
    fs : int
        Sample frequency in Hz.
    """
    import soundfile as sf
    import os

    # Try with and without extension
    if not os.path.isfile(filename):
        if os.path.isfile(filename + '.aif'):
            filename = filename + '.aif'
        elif os.path.isfile(filename + '.aiff'):
            filename = filename + '.aiff'
        else:
            raise FileNotFoundError(f"Cannot open {filename} for input")

    info = sf.info(filename)
    fs = info.samplerate
    total_samples = info.frames

    start = nskip
    if nmax >= 0:
        stop = min(start + nmax, total_samples)
    else:
        stop = total_samples

    # Determine scaling
    if not mode:
        mode = 'p'
    sc = 'p'
    for c in mode:
        if c in 'prsq':
            sc = c
            break

    if sc == 'r':
        # Read as integer
        y, _ = sf.read(filename, start=start, stop=stop, dtype='int32')
    else:
        # Read as float (normalized to [-1, 1])
        y, _ = sf.read(filename, start=start, stop=stop, dtype='float64')

        if sc == 's':
            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak

    if y.ndim == 1:
        y = y[:, np.newaxis]

    return y, fs
