"""V_WRITEHTK - Write data in HTK format."""

import struct
import numpy as np


def v_writehtk(file, d, fp, tc):
    """Write data in HTK format.

    Parameters
    ----------
    file : str
        Path to the output file.
    d : array_like
        Data to write: one row per frame.
    fp : float
        Frame period in seconds.
    tc : int
        Type code (see v_readhtk for details).
    """
    d = np.asarray(d, dtype=np.float64)

    # Silently ignore checksum request: clear bit 13 (4096)
    tc = tc & ~4096

    if d.ndim == 1:
        d = d.reshape(-1, 1)

    nf, nv = d.shape

    # Extract bits from type code
    nhb = 10
    ndt = 6
    hb = np.zeros(nhb + 1, dtype=int)
    for i in range(nhb + 1):
        hb[i] = int(np.floor(tc * 2.0 ** (-(ndt + nhb) + i)))
    hd = np.zeros(nhb, dtype=int)
    for i in range(nhb):
        hd[i] = hb[nhb - i] - 2 * hb[nhb - 1 - i]
    dt = tc - int(hb[-1]) * (2 ** ndt)

    # If waveform is a row vector (nf==1, dt==0), treat as column
    if dt == 0 and nf == 1:
        d = d.T
        nf, nv = d.shape

    # Handle compression
    if hd[4]:  # compressed
        dx = np.max(d, axis=0)
        dn = np.min(d, axis=0)
        a = np.ones(nv)
        b = dx.copy()
        mk = dx > dn
        a[mk] = 65534.0 / (dx[mk] - dn[mk])
        b[mk] = 0.5 * (dx[mk] + dn[mk]) * a[mk]
        d = d * a[np.newaxis, :] - b[np.newaxis, :]
        nf = nf + 4

    # Adjust tc for int16 storage
    tc_write = tc
    if tc_write > 32767:
        tc_write = tc_write - 65536

    with open(file, 'wb') as f:
        # Write header
        f.write(struct.pack('>i', nf))
        f.write(struct.pack('>i', round(fp * 1e7)))

        if dt in (0, 5, 10) or hd[4]:
            # Write data as int16
            if dt == 5:
                d = d * 32767
                if hd[4]:
                    raise ValueError('Cannot use compression with IREFC format')

            nby = nv * 2
            if nby > 32767:
                raise ValueError(
                    f'byte count of frame is {nby} which exceeds 32767 '
                    '(is data transposed?)')

            f.write(struct.pack('>h', nby))
            f.write(struct.pack('>h', tc_write))

            if hd[4]:
                # Write compression factors
                for val in a:
                    f.write(struct.pack('>f', val))
                for val in b:
                    f.write(struct.pack('>f', val))

            # Write data row by row (transposed for column-major order)
            int_data = np.round(d).astype(np.int16)
            # Write in row-major order (each row is a frame)
            f.write(int_data.astype('>i2').tobytes())
        else:
            # Write data as float32
            nby = nv * 4
            if nby > 32767:
                raise ValueError(
                    f'byte count of frame is {nby} which exceeds 32767 '
                    '(is data transposed?)')

            f.write(struct.pack('>h', nby))
            f.write(struct.pack('>h', tc_write))

            # Write data
            float_data = d.astype(np.float32)
            f.write(float_data.astype('>f4').tobytes())
