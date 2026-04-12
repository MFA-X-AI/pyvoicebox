"""V_READSFS - Read a .SFS (Speech Filing System) format sound file.

The SFS format was developed by Mark Huckvale at UCL for speech research.
This is a simplified Python reader for the most common data types.
"""

from __future__ import annotations
import numpy as np
import struct
import os


def _zerotrim(data):
    """Remove trailing zero bytes and convert to string."""
    idx = data.find(b'\x00')
    if idx >= 0:
        data = data[:idx]
    return data.decode('ascii', errors='replace').strip()


def v_readsfs(filename, ty=1, sub=-1, mode='p', nmax=-1, nskip=0) -> tuple[np.ndarray, float, dict]:
    """Read a .SFS format sound file.

    Parameters
    ----------
    filename : str
        Path to the SFS file.
    ty : int, optional
        Type of data item: 0=main header, 1=speech, 2=laryngograph,
        5=annotation. Default is 1.
    sub : int, optional
        Instance of type ty: 0=first, -1=last (default).
    mode : str, optional
        Mode string. Default is 'p'.
        'i' : Force integer data to be at least 16 bits.
    nmax : int, optional
        Maximum number of samples to read. -1 for unlimited.
    nskip : int, optional
        Number of samples to skip from start.

    Returns
    -------
    y : ndarray
        Data array. For speech data, column vector.
    fs : float
        Sample frequency in Hz.
    hd : dict
        Header information.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Cannot open {filename} for input")

    with open(filename, 'rb') as fid:
        # Read main header
        t = fid.read(512)
        if len(t) < 512:
            raise IOError(f"Cannot read header from SFS file {filename}")

        if t[:3] != b'UC2':
            raise ValueError(f"{filename} is not an SFS file type UC2")

        byte_order = t[511]  # byte order indicator
        bo = '>' if byte_order == 0 else '<'

        # Read item list
        itemlist = [(0, 1, 0, 0, byte_order)]
        proglist = [('', '', '')]

        for i in range(1, 200):
            pos = fid.tell()
            t = fid.read(512)
            if len(t) < 512:
                break

            item_bo = t[511]
            ibo = '>' if item_bo == 0 else '<'

            # Parse item header
            item_type = struct.unpack(ibo + 'i', t[388:392])[0]
            item_subtype = struct.unpack(ibo + 'i', t[392:396])[0]
            item_length = struct.unpack(ibo + 'i', t[412:416])[0]

            if abs(item_type) > 29:
                break

            itemlist.append((item_type, item_subtype, item_length, pos, item_bo))
            proglist.append((
                _zerotrim(t[0:256]),
                _zerotrim(t[256:384]),
                _zerotrim(t[436:456]),
            ))

            # Skip data
            fid.seek(item_length, 1)

        # Find requested item
        it = None
        if ty == 0:
            it = 0
        else:
            matches = [i for i, item in enumerate(itemlist) if item[0] == ty]
            if not matches:
                raise ValueError(f"Cannot find item type {ty} in file {filename}")
            if sub == 0:
                it = matches[0]
            elif sub == -1:
                it = matches[-1]
            else:
                matches_sub = [i for i in matches if itemlist[i][1] == sub]
                if matches_sub:
                    it = matches_sub[0]
                else:
                    raise ValueError(f"Cannot find item {ty}.{sub} in file {filename}")

        y = np.array([])
        fs = 0.0
        hd = {}

        if it == 0:
            # Read main header info
            fid.seek(0)
            mb = fid.read(512)
            hd['file_type'] = _zerotrim(mb[0:4])
            return y, fs, hd

        # Read item data
        lit = itemlist[it]
        item_bo = lit[4]
        ibo = '>' if item_bo == 0 else '<'

        fid.seek(lit[3])  # seek to item header
        # Read the 512-byte item header
        ihdr = fid.read(512)

        # Parse item header fields
        processing_history = _zerotrim(ihdr[0:256])
        parameters = _zerotrim(ihdr[256:384])

        # Read numeric header fields
        hdr = np.zeros(14)
        hdr_bytes = ihdr[384:384 + 32]
        for j in range(8):
            hdr[j] = struct.unpack(ibo + 'i', hdr_bytes[j * 4:(j + 1) * 4])[0]

        # Frame duration (double)
        hdr[8] = struct.unpack(ibo + 'd', ihdr[416:424])[0]
        if hdr[8] > 0:
            fs = 1.0 / hdr[8]

        # Data present
        hdr[9] = struct.unpack(ibo + 'i', ihdr[424:428])[0]

        # Time offset (double)
        hdr[10] = struct.unpack(ibo + 'd', ihdr[428:436])[0]

        comment = _zerotrim(ihdr[436:456])

        # Remaining fields
        remaining = ihdr[456:468]
        if len(remaining) >= 12:
            for j in range(3):
                hdr[11 + j] = struct.unpack(ibo + 'i', remaining[j * 4:(j + 1) * 4])[0]

        hd = {
            'processing_history': processing_history,
            'parameters': parameters,
            'comment': comment,
            'data_type': int(hdr[1]),
            'subtype': int(hdr[2]),
            'floating': int(hdr[3]),
            'datasize': int(hdr[4]),
            'framesize': int(hdr[5]),
            'numframes': int(hdr[6]),
            'data_length': int(hdr[7]),
            'frame_duration': hdr[8],
            'data_present': int(hdr[9]),
            'time_offset': hdr[10],
            'sample_rate': fs,
        }

        # Read data
        ksamples = int(hdr[6]) - nskip
        if nmax >= 0:
            ksamples = min(nmax, ksamples)

        if ksamples > 0 and int(hdr[9]) == 1:
            ds = int(hdr[4])  # data size in bytes
            fsz = int(hdr[5])  # frame size

            if int(hdr[3]) >= 0:  # non-structured
                if int(hdr[3]) > 0:  # floating point
                    if ds == 4:
                        dtype = ibo + 'f'
                        np_dtype = np.float32
                    elif ds == 8:
                        dtype = ibo + 'd'
                        np_dtype = np.float64
                    else:
                        raise ValueError("Invalid data size in SFS file")
                else:  # integer
                    if ds == 1 and 'i' not in mode:
                        np_dtype = np.uint8
                    elif ds <= 2:
                        np_dtype = np.dtype(ibo + 'i2')
                        fsz = int(np.ceil(fsz * ds / 2))
                    elif ds == 4:
                        np_dtype = np.dtype(ibo + 'i4')
                    else:
                        raise ValueError("Invalid data size in SFS file")

                # Seek to data start
                fid.seek(lit[3] + 512 + nskip * fsz * ds)
                nd = fsz * ksamples
                raw = fid.read(nd * ds)
                y = np.frombuffer(raw, dtype=np_dtype, count=nd)
                y = y.astype(float)
                if fsz > 1:
                    y = y.reshape(ksamples, fsz)
                else:
                    y = y[:, np.newaxis]

    return y, fs, hd
