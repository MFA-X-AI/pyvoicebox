"""V_RANDISCR - Generate discrete random values with specified probabilities."""

import numpy as np


def v_randiscr(p=None, n=1, a=None):
    """Generate discrete random numbers with specified probabilities.

    Parameters
    ----------
    p : array_like, optional
        Probabilities (not necessarily normalized). Default: uniform.
    n : int, optional
        Number of random values to generate. Positive = with replacement,
        negative = without replacement. Default: 1.
    a : array_like, optional
        Output alphabet. Default: 0-based indices.

    Returns
    -------
    x : ndarray
        Vector of values taken from alphabet a.
    """
    got_a = a is not None

    if p is None or (isinstance(p, np.ndarray) and p.size == 0):
        if got_a:
            a = np.asarray(a)
            p = np.ones(len(a))
        else:
            p = np.ones(2)
            a = np.arange(2)
            got_a = True

    p = np.asarray(p, dtype=float)
    q = p.ravel()
    d = len(q)

    if got_a:
        a = np.asarray(a)
        if d != a.size:
            raise ValueError('sizes of alphabet and probability vector must match')

    if n >= 1:
        # Sample with replacement
        n = int(n)
        cs = np.cumsum(q / np.sum(q))
        cs[-1] = 1.0  # ensure no numerical issues
        r = np.random.rand(n)
        x = np.searchsorted(cs, r)
        # 0-based indices
    else:
        # Sample without replacement
        n = abs(int(n))
        if n > d:
            raise ValueError('Number of output samples exceeds alphabet size')
        if np.all(q == q[0]):
            # Uniform probabilities
            perm = np.random.permutation(d)
            x = perm[:n]
        else:
            cs = np.cumsum(q / np.sum(q))
            cs[-1] = 1.0
            chosen = set()
            x_list = []
            while len(x_list) < n:
                r = np.random.rand()
                idx = np.searchsorted(cs, r)
                if idx not in chosen:
                    chosen.add(idx)
                    x_list.append(idx)
            x = np.array(x_list)

    if got_a:
        x = a.ravel()[x]
    return x
