"""V_READSPH - Read a SPHERE/TIMIT format sound file.

The SPHERE (SPeech HEader REsources) format is used by NIST for
speech corpora like TIMIT.
"""

import numpy as np
import struct
import os


def v_readsph(filename, mode='p', nmax=-1, nskip=0):
    """Read a SPHERE/TIMIT format sound file.

    Parameters
    ----------
    filename : str
        Path to the SPH file (with or without .sph extension).
    mode : str, optional
        Scaling/format mode string. Default is 'p'.
        'p' : Scaled so +-1 equals full scale (default).
        'r' : Raw unscaled data (integer values).
        's' : Auto scale to make data peak = +-1.
        'l' : Force little endian byte order.
        'b' : Force big endian byte order.
        'w' : Also read .wrd annotation file.
        't' : Also read .phn phonetic transcription file.
    nmax : int, optional
        Maximum number of samples to read. -1 for unlimited.
    nskip : int, optional
        Number of samples to skip from start.

    Returns
    -------
    y : ndarray
        Data matrix of shape (samples, channels).
    fs : int
        Sample frequency in Hz.
    wrd : list of tuple, optional
        Word annotations [(start_time, end_time, text), ...].
        Only returned if 'w' in mode.
    phn : list of tuple, optional
        Phoneme annotations [(start_time, end_time, text), ...].
        Only returned if 't' in mode.
    ffx : dict
        File information dictionary.
    """
    if not mode:
        mode = 'p'

    # Determine scaling mode
    sc = 'p'
    for c in mode:
        if c in 'prs':
            sc = c
            break

    # Find file
    if not os.path.isfile(filename):
        if os.path.isfile(filename + '.sph'):
            filename = filename + '.sph'
        elif os.path.isfile(filename + '.wav'):
            filename = filename + '.wav'
        else:
            raise FileNotFoundError(f"Cannot open {filename} for input")

    # Default byte order
    byte_order = 'little'
    if 'l' in mode:
        byte_order = 'little'
    elif 'b' in mode:
        byte_order = 'big'

    bo = '<' if byte_order == 'little' else '>'

    with open(filename, 'rb') as fid:
        # Read header
        first_line = fid.read(16)
        if len(first_line) < 16:
            raise IOError("File does not begin with a SPHERE header")

        fmt = first_line[:7].decode('ascii', errors='replace').strip()
        try:
            hlen = int(first_line[8:15].decode('ascii').strip())
        except ValueError:
            raise IOError("File does not begin with a SPHERE header")

        # Parse header fields
        hdr = {}
        while True:
            line = b''
            while True:
                ch = fid.read(1)
                if not ch or ch == b'\n':
                    break
                line += ch
            line_str = line.decode('ascii', errors='replace').strip()

            if not line_str or line_str.startswith(';'):
                if line_str.startswith('end_head'):
                    break
                continue

            parts = line_str.split(None, 2)
            if len(parts) < 2:
                if 'end_head' in line_str:
                    break
                continue

            field_name = parts[0]
            type_spec = parts[1]

            if len(parts) >= 3:
                value_str = parts[2]
            else:
                value_str = ''

            if type_spec.startswith('-s'):
                # String type
                try:
                    slen = int(type_spec[2:])
                    hdr[field_name] = value_str[:slen].strip()
                except ValueError:
                    hdr[field_name] = value_str.strip()
            elif type_spec.startswith('-i'):
                try:
                    hdr[field_name] = int(value_str)
                except ValueError:
                    hdr[field_name] = 0
            elif type_spec.startswith('-r'):
                try:
                    hdr[field_name] = float(value_str)
                except ValueError:
                    hdr[field_name] = 0.0
            else:
                hdr[field_name] = value_str

        # Determine byte order from header
        if 'sample_byte_format' in hdr:
            sbf = hdr['sample_byte_format']
            if sbf.startswith('0'):
                byte_order = 'little'
            else:
                byte_order = 'big'
            if 'l' not in mode and 'b' not in mode:
                bo = '<' if byte_order == 'little' else '>'

        # Extract key parameters
        nsamp = hdr.get('sample_count', 0)
        nchan = hdr.get('channel_count', 1)
        nbytes = hdr.get('sample_n_bytes', 2)
        nbits = hdr.get('sample_sig_bits', 16)
        fs = hdr.get('sample_rate', 1)

        # Determine coding
        coding = hdr.get('sample_coding', 'pcm')
        is_ulaw = 'ulaw' in coding.lower() or 'mu-law' in coding.lower()

        if nsamp == 0:
            fid.seek(0, 2)
            file_size = fid.tell()
            nsamp = (file_size - hlen) // (nchan * nbytes)

        # Read data
        start = nskip
        ksamples = nsamp - start
        if nmax >= 0:
            ksamples = min(nmax, ksamples)

        if ksamples > 0:
            fid.seek(hlen + nchan * nbytes * start)
            nread = nchan * ksamples

            if nbytes == 1:
                raw = np.frombuffer(fid.read(nread), dtype=np.uint8)
                if is_ulaw:
                    from pyvoicebox.v_pcmu2lin import v_pcmu2lin
                    y = v_pcmu2lin(raw.astype(float))
                    pk = 2.005649
                else:
                    y = raw.astype(float) - 128
                    pk = 128
            elif nbytes == 2:
                dtype = np.dtype(bo + 'i2')
                y = np.frombuffer(fid.read(nread * 2), dtype=dtype).astype(float)
                pk = 32768
            elif nbytes == 4:
                dtype = np.dtype(bo + 'i4')
                y = np.frombuffer(fid.read(nread * 4), dtype=dtype).astype(float)
                pk = 2**31
            else:
                raise ValueError(f"Unsupported sample size: {nbytes} bytes")

            # Scale
            if sc == 's':
                peak = np.max(np.abs(y))
                if peak > 0:
                    y = y / peak
            elif sc == 'p':
                if not is_ulaw or nbytes > 1:
                    y = y / pk
            # 'r' mode: no scaling

            if nchan > 1:
                y = y.reshape(-1, nchan)
            else:
                y = y[:, np.newaxis]
        else:
            y = np.array([]).reshape(0, nchan)

    ffx = {
        'filename': filename,
        'header': hdr,
        'format': fmt,
        'sample_count': nsamp,
        'channel_count': nchan,
        'sample_n_bytes': nbytes,
        'sample_sig_bits': nbits,
        'sample_rate': fs,
    }

    # Read annotation files if requested
    results = [y, fs]

    if 'w' in mode:
        wrd = _read_annotation(filename, 'wrd', fs)
        results.append(wrd)

    if 't' in mode:
        phn = _read_annotation(filename, 'phn', fs)
        results.append(phn)

    results.append(ffx)
    return tuple(results)


def _read_annotation(filename, ext, fs):
    """Read a TIMIT-style annotation file (.wrd or .phn).

    Parameters
    ----------
    filename : str
        Path to the speech file.
    ext : str
        Extension of the annotation file ('wrd' or 'phn').
    fs : int
        Sample rate for converting sample indices to times.

    Returns
    -------
    annotations : list of tuple
        List of (start_time, end_time, text) tuples.
    """
    base = os.path.splitext(filename)[0]
    ann_file = base + '.' + ext
    annotations = []

    if os.path.isfile(ann_file):
        with open(ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 2)
                if len(parts) >= 3:
                    try:
                        start = int(parts[0]) / fs
                        end = int(parts[1]) / fs
                        text = parts[2].strip()
                        annotations.append((start, end, text))
                    except ValueError:
                        continue

    return annotations
