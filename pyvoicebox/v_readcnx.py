"""V_READCNX - Read a .CNX format sound file.

This is the format of the BT Connex-S1 alphabet database.
"""

from __future__ import annotations
import numpy as np
import struct
import os


def v_readcnx(filename, mode='') -> tuple[np.ndarray, float, dict]:
    """Read a .CNX format sound file.

    Parameters
    ----------
    filename : str
        Path to the CNX file (with or without .cnx extension).
    mode : str, optional
        Mode string:
        't' : trim to start/end samples indicated in header
        'h' : read header only

    Returns
    -------
    y : ndarray
        Column vector containing the waveform (int16 samples).
    fs : float
        Sample frequency in Hz.
    h : dict
        Header parameters:
        'num_samples' : number of samples in file
        'status' : 0=good, 1=bad
        'start_sample' : start sample number
        'end_sample' : ending sample number
        'speaker_id' : speaker identification number
        'speaker_age' : speaker age group
        'speaker_sex' : 0=male, 1=female
        'ascii_char' : ascii character
        'repetition' : repetition number
    """
    if not os.path.isfile(filename):
        if os.path.isfile(filename + '.cnx'):
            filename = filename + '.cnx'
        else:
            raise FileNotFoundError(f"Cannot open {filename} for input")

    # Field index table (0-indexed delimiters)
    # Original MATLAB ix:
    # ix=[17 71; 18 0; 19 0; 10 0; 12 0; 13 77; 15 -1; 16 0]
    # Columns: [delimiter_index, special_value]
    ix = [
        (16, 71),   # status: check against 'G' (71)
        (17, 0),    # start_sample
        (18, 0),    # end_sample
        (9, 0),     # speaker_id
        (11, 0),    # speaker_age
        (12, 77),   # speaker_sex: check against 'M' (77)
        (14, -1),   # ascii_char: raw byte
        (15, 0),    # repetition
    ]

    with open(filename, 'rb') as fid:
        hdr = fid.read(512)
        if len(hdr) != 512:
            raise IOError(f"Cannot read header from connex file {filename}")

        # Find delimiters (pipe characters) starting from byte 4
        delimiters = []
        for i in range(4, len(hdr)):
            if hdr[i] == ord('|'):
                delimiters.append(i)

        # Parse sample frequency from first field
        # Characters from byte 16 to first delimiter
        fs_str = hdr[16:delimiters[0]].decode('ascii', errors='replace').strip()
        fs = float(fs_str)

        h = {}
        field_names = ['status', 'start_sample', 'end_sample', 'speaker_id',
                       'speaker_age', 'speaker_sex', 'ascii_char', 'repetition']

        for idx, (del_idx, special) in enumerate(ix):
            # Get field between delimiters
            start = delimiters[del_idx - 1] + 1 if del_idx > 0 else 4
            end = delimiters[del_idx] if del_idx < len(delimiters) else len(hdr)
            field_bytes = hdr[start:end]

            # Find '=' sign
            eq_pos = field_bytes.find(b'=')
            if eq_pos < 0:
                h[field_names[idx]] = 0
                continue

            value_bytes = field_bytes[eq_pos + 1:]

            if special == -1:
                # Raw byte value
                h[field_names[idx]] = value_bytes[0] if len(value_bytes) > 0 else 0
            elif special > 0:
                # Check against character
                ch = value_bytes[0] if len(value_bytes) > 0 else 0
                h[field_names[idx]] = 0 if ch == special else 1
            else:
                # Numeric value
                try:
                    val_str = value_bytes.decode('ascii', errors='replace').strip()
                    h[field_names[idx]] = int(val_str) if val_str else 0
                except (ValueError, IndexError):
                    h[field_names[idx]] = 0

        if 'h' in mode:
            return np.array([]), fs, h

        if 't' in mode:
            # Read trimmed data
            start_samp = h.get('start_sample', 0)
            end_samp = h.get('end_sample', 0)
            fid.seek(512 + 2 * start_samp, 0)  # skip to status offset, not start_sample
            count = end_samp - start_samp + 1
            y = np.frombuffer(fid.read(count * 2), dtype='<i2')
        else:
            y = np.frombuffer(fid.read(), dtype='<i2')

        # Compute total samples
        fid.seek(0, 2)
        total_bytes = fid.tell()
        h['num_samples'] = (total_bytes - 512) // 2

    y = y.astype(float)
    if y.ndim == 1:
        y = y[:, np.newaxis]

    return y, fs, h
