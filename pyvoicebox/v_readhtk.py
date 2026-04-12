"""V_READHTK - Read an HTK parameter file."""

from __future__ import annotations
import struct
import numpy as np


# Data type names
_HTK_KINDS = [
    'WAVEFORM', 'LPC', 'LPREFC', 'LPCEPSTRA', 'LPDELCEP', 'IREFC',
    'MFCC', 'FBANK', 'MELSPEC', 'USER', 'DISCRETE', 'PLP', 'ANON'
]

# Suffix codes
_HTK_SUFFIXES = 'ENDACZK0VT'


def v_readhtk(file) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Read an HTK parameter file.

    Parameters
    ----------
    file : str
        Path to the HTK file.

    Returns
    -------
    d : ndarray
        Data: column vector for waveforms, one row per frame for other types.
    fp : float
        Frame period in seconds.
    dt : int
        Base data type (0-12).
    tc : int
        Full type code including modifiers.
    t : str
        Text version of type code, e.g. 'LPC_C_K'.
    """
    with open(file, 'rb') as f:
        # Read header (12 bytes)
        nf = struct.unpack('>i', f.read(4))[0]       # number of frames
        fp = struct.unpack('>i', f.read(4))[0] * 1e-7  # frame period in seconds
        by = struct.unpack('>h', f.read(2))[0]       # bytes per frame
        tc = struct.unpack('>h', f.read(2))[0]       # type code

        # Handle negative tc (unsigned interpretation)
        if tc < 0:
            tc = tc + 65536

        # Extract suffix bits and base data type
        cc = _HTK_SUFFIXES
        nhb = len(cc)
        ndt = 6

        # Extract bits from type code
        hb = np.zeros(nhb + 1, dtype=int)
        for i in range(nhb + 1):
            hb[i] = int(np.floor(tc * 2.0 ** (-(ndt + nhb) + i)))
        hd = np.zeros(nhb, dtype=int)
        for i in range(nhb):
            hd[i] = hb[nhb - i] - 2 * hb[nhb - 1 - i]
        dt = tc - int(hb[-1]) * (2 ** ndt)

        # Handle IREFC hack
        if dt == 5:
            # Check file length to detect if stored as compressed LPREFC
            pos = f.tell()
            f.seek(0, 2)
            flen = f.tell()
            f.seek(12, 0)
            if flen > 14 + by * nf:
                dt = 2
                hd[4] = 1  # set compressed flag (index 4 = _C)
                nf = nf + 4

        if dt in (0, 5, 10):
            # 16-bit data for waveforms, IREFC and DISCRETE
            ncol = by // 2
            raw = np.frombuffer(f.read(nf * ncol * 2), dtype='>i2')
            d = raw.reshape(nf, ncol).astype(np.float64)
            if dt == 5:
                d = d / 32767.0
        else:
            if hd[4]:  # compressed data
                nf = nf - 4
                ncol = by // 2
                scales = np.frombuffer(f.read(ncol * 4), dtype='>f4').astype(np.float64)
                biases = np.frombuffer(f.read(ncol * 4), dtype='>f4').astype(np.float64)
                raw = np.frombuffer(f.read(nf * ncol * 2), dtype='>i2').astype(np.float64)
                raw = raw.reshape(nf, ncol)
                d = (raw + biases[np.newaxis, :]) / scales[np.newaxis, :]
            else:
                # Uncompressed float data
                ncol = by // 4
                raw = np.frombuffer(f.read(nf * ncol * 4), dtype='>f4')
                d = raw.reshape(nf, ncol).astype(np.float64)

    # Build text type string
    kind_idx = min(dt, len(_HTK_KINDS) - 1)
    t = _HTK_KINDS[kind_idx]
    for i in range(nhb):
        if hd[i] > 0:
            t += '_' + cc[i]

    return d, fp, dt, tc, t
