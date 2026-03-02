"""V_READWAV - Read a .WAV format sound file.

Uses the soundfile library for core WAV I/O, preserving the MATLAB
function signature for compatibility.
"""

import numpy as np
import soundfile as sf


def v_readwav(filename, mode='p', nmax=-1, nskip=0):
    """Read a .WAV format sound file.

    Parameters
    ----------
    filename : str
        Path to the WAV file (with or without .wav extension).
    mode : str, optional
        Scaling mode string. Default is 'p'.
            'p' : Scaled so +-1 equals full scale (default).
            'r' : Raw unscaled data (integer values).
            's' : Auto scale to make data peak = +-1.
            'q' : Scaled to make 0dBm0 be unity mean square.
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
    info = sf.info(filename)
    fs = info.samplerate
    nchannels = info.channels
    subtype = info.subtype

    # Determine how many samples to read
    total_samples = info.frames
    start = nskip
    if nmax >= 0:
        stop = min(start + nmax, total_samples)
    else:
        stop = total_samples

    # Determine scaling mode
    if not mode:
        mode = 'p'
    # Find the first scaling character
    sc = 'p'
    for c in mode:
        if c in 'prsq':
            sc = c
            break

    # Read as float (soundfile default gives +-1 range for integer formats)
    if sc == 'r':
        # For raw mode, read as integer values
        # Determine dtype based on subtype
        if 'PCM_8' in subtype:
            dtype = 'int16'  # soundfile doesn't support int8, we'll handle
        elif 'PCM_16' in subtype:
            dtype = 'int16'
        elif 'PCM_24' in subtype:
            dtype = 'int32'
        elif 'PCM_32' in subtype:
            dtype = 'int32'
        elif 'FLOAT' in subtype or 'DOUBLE' in subtype:
            dtype = 'float64'
        else:
            dtype = 'float64'

        if 'FLOAT' in subtype or 'DOUBLE' in subtype:
            y, _ = sf.read(filename, start=start, stop=stop, dtype='float64',
                           always_2d=True)
        else:
            y, _ = sf.read(filename, start=start, stop=stop, dtype=dtype,
                           always_2d=True)
            y = y.astype(np.float64)
    else:
        # Read as float64, soundfile normalizes to +-1
        y, _ = sf.read(filename, start=start, stop=stop, dtype='float64',
                       always_2d=True)

        if sc == 's':
            # Auto scale to peak = +-1
            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak
        elif sc == 'q':
            # Scale to 0dBm0 (ITU G.711)
            # For mu-law format use 2.03761563, else use 2.03033976
            # Since soundfile normalizes to +-1, we just multiply
            # We'd need format info to know if mu-law, default to A-law factor
            y = y * 2.03033976
        # 'p' mode: already +-1 from soundfile

    # If only one channel, still return 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return y, fs
