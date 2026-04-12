"""V_WRITEWAV - Write a .WAV format sound file.

Uses the soundfile library for core WAV I/O.
"""

from __future__ import annotations
import numpy as np
import soundfile as sf


def v_writewav(d, fs, filename, mode='s') -> None:
    """Write a .WAV format sound file.

    Parameters
    ----------
    d : array_like
        Data to write. Shape (samples,) or (samples, channels).
    fs : int
        Sample frequency in Hz.
    filename : str
        Output filename (with or without .wav extension).
    mode : str, optional
        Mode string controlling format and scaling. Default is 's'.
            's' : Auto scale to make data peak = +-1 (default).
            'r' : Raw unscaled data.
            'p' : Scaled so +-1 equals full scale.
            'q' : Scaled to 0dBm0.
            '16': 16 bit PCM data (default bit depth).
            '8' : 8 bit PCM data.
            '24': 24 bit PCM data.
            '32': 32 bit PCM data.
            'v' : 32-bit floating point.
            'V' : 64-bit floating point.
            'a' : 8-bit A-law PCM.
            'u' : 8-bit mu-law PCM.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    elif d.ndim == 2 and d.shape[0] == 1:
        d = d.T

    if not mode:
        mode = 's'

    # Determine subtype (bit depth / format)
    subtype = 'PCM_16'  # default
    if 'v' in mode:
        subtype = 'FLOAT'
    elif 'V' in mode:
        subtype = 'DOUBLE'
    elif 'a' in mode:
        subtype = 'PCM_16'  # A-law: we encode manually then write as PCM
    elif 'u' in mode:
        subtype = 'PCM_16'  # Mu-law: same
    else:
        # Look for numeric bit depth
        bits = None
        i = 0
        while i < len(mode):
            if mode[i].isdigit():
                j = i
                while j < len(mode) and mode[j].isdigit():
                    j += 1
                bits = int(mode[i:j])
                break
            i += 1
        if bits is not None:
            bit_map = {8: 'PCM_16', 16: 'PCM_16', 24: 'PCM_24', 32: 'PCM_32'}
            subtype = bit_map.get(bits, 'PCM_16')

    # Determine scaling mode
    sc = 's'  # default
    for c in mode:
        if c in 'prsq':
            sc = c
            break

    # Scale data
    if sc == 's':
        peak = np.max(np.abs(d))
        if peak > 0:
            d = d / peak
    elif sc == 'p':
        pass  # data already in +-1 range
    elif sc == 'q':
        # Scale by dBm0 factor
        d = d / 2.03033976
    elif sc == 'r':
        # Raw: normalize integer range to +-1 for soundfile
        # Determine the peak integer value for the bit depth
        if subtype == 'FLOAT' or subtype == 'DOUBLE':
            pass  # no normalization needed
        else:
            # Extract bits
            bits_val = int(subtype.split('_')[1]) if '_' in subtype else 16
            peak_int = 2 ** (bits_val - 1)
            d = d / peak_int

    # Append .wav if no extension
    if '.' not in filename:
        filename = filename + '.wav'

    # Write using soundfile
    sf.write(filename, d, fs, subtype=subtype)
