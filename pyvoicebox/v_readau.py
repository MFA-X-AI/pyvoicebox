"""V_READAU - Read a SUN .AU format sound file.

Uses the soundfile library for robust AU file reading.
"""

import numpy as np


def v_readau(filename, mode=''):
    """Read a SUN .AU format sound file.

    Parameters
    ----------
    filename : str
        Path to the AU file (with or without .au extension).
    mode : str, optional
        Mode string:
        't' : trim leading and trailing silences
        'h' : read header only

    Returns
    -------
    y : ndarray
        Data matrix with one channel per column.
        If mode='h', returns header parameters as a dict.
    fs : int
        Sample frequency in Hz.
    h : dict
        Header parameters:
        'header_length', 'data_length', 'data_format',
        'sample_rate', 'num_channels'.
    """
    import soundfile as sf
    import os

    if not os.path.isfile(filename):
        if os.path.isfile(filename + '.au'):
            filename = filename + '.au'
        else:
            raise FileNotFoundError(f"Cannot open {filename} for input")

    info = sf.info(filename)
    fs = info.samplerate
    h = {
        'sample_rate': info.samplerate,
        'num_channels': info.channels,
        'frames': info.frames,
        'format': info.format,
        'subtype': info.subtype,
    }

    if 'h' in mode:
        return h, fs, h

    y, _ = sf.read(filename, dtype='float64')

    if y.ndim == 1:
        y = y[:, np.newaxis]

    if 't' in mode:
        # Trim leading and trailing silence
        energy = np.sum(y ** 2, axis=1)
        threshold = np.max(energy) * 1e-4
        nonsilent = np.where(energy > threshold)[0]
        if len(nonsilent) > 0:
            y = y[nonsilent[0]:nonsilent[-1] + 1, :]

    return y, fs, h
