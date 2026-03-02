"""V_FRAC2BIN - Convert a column vector to binary string representation."""

import numpy as np


def v_frac2bin(d, n=1, m=0):
    """Convert a column vector to binary S=(D,N,M).

    Parameters
    ----------
    d : array_like
        Scalar or 1-D array of values to convert.
    n : int, optional
        Minimum number of integer bits to output. If negative, leading zeros
        will be output as spaces for positions to the left of |n|'th integer
        column. Default: 1.
    m : int, optional
        Number of places after binary point. If negative, values are truncated
        rather than rounded. Default: 0.

    Returns
    -------
    s : list of str
        List of binary string representations, one per input value.
    """
    d = np.atleast_1d(np.asarray(d, dtype=float))
    l = abs(n)
    r = abs(m)

    # Find the maximum value's exponent
    max_val = np.max(d)
    if max_val > 0:
        _, e = np.frexp(max_val)  # e such that max_val = f * 2^e, 0.5 <= f < 1
    else:
        e = 0

    # Compute scaled values
    if m < 0:
        v = np.floor(d * (2.0 ** r))
    else:
        v = np.round(d * (2.0 ** r))

    v = v.astype(int)

    # Total number of bits needed: max(l, e) + r
    total_bits = max(l, e) + r

    # Generate binary strings
    result = []
    for val in v:
        if total_bits <= 0:
            bits = '0'
        else:
            bits = ''
            for bit_pos in range(total_bits - 1, -1, -1):
                bits += '1' if (val >> bit_pos) & 1 else '0'

        # Insert binary point if r > 0
        if r > 0:
            int_part = bits[:len(bits) - r]
            frac_part = bits[len(bits) - r:]
            if not int_part:
                int_part = '0'
            s = int_part + '.' + frac_part
        else:
            s = bits

        # Handle leading zeros -> spaces when n < 0
        if n < 0:
            bp = s.find('.')
            if bp < 0:
                bp = len(s)
            # Replace leading zeros with spaces for positions to the left of l'th column from point
            for i in range(bp - l):
                if s[i] == '0':
                    s = s[:i] + ' ' + s[i + 1:]
                else:
                    break

        result.append(s)

    return result
