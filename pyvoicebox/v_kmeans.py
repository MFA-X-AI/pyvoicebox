"""V_KMEANS - K-means clustering algorithm."""

import numpy as np
from .v_disteusq import v_disteusq
from .v_rnsubset import v_rnsubset


def v_kmeans(d, k, x0='f', l=300):
    """Vector quantization using K-means algorithm.

    Parameters
    ----------
    d : array_like
        Data vectors, shape (n, p).
    k : int
        Number of centres required.
    x0 : str or array_like, optional
        Initial centres (k, p) or initialization method:
        'f' = pick k random data points (default),
        'p' = random partition centroids.
    l : int, optional
        Maximum number of iterations (default 300). Use 0 to just compute
        distances for given centres.

    Returns
    -------
    x : ndarray
        Output centres (k, p), or mean squared error if l=0.
    g : float or ndarray
        Mean squared error, or cluster assignments if l=0.
    j : ndarray
        Cluster assignments for each data point (0-based).
    gg : ndarray
        Mean squared error at each iteration (only if l > 0).
    """
    d = np.asarray(d, dtype=float)
    n, p = d.shape

    if isinstance(x0, str):
        if k < n:
            if 'p' in x0:
                # Random partition initialization
                ix = np.random.randint(0, k, size=n)
                forced = v_rnsubset(k, n)
                ix[forced] = np.arange(k)
                x = np.zeros((k, p))
                for i in range(k):
                    mask = ix == i
                    if np.any(mask):
                        x[i, :] = np.mean(d[mask, :], axis=0)
            else:
                # Forgy initialization: sample k centres
                x = d[v_rnsubset(k, n), :]
        else:
            x = d[np.arange(k) % n, :]
    else:
        x = np.asarray(x0, dtype=float).copy()

    m = np.zeros(n)
    j = np.zeros(n, dtype=int)
    gg = np.zeros(l)

    if l > 0:
        for ll in range(l):
            # Find closest centre
            z = v_disteusq(d, x, 'x')
            j = np.argmin(z, axis=1)
            m = z[np.arange(n), j]

            y = x.copy()

            # Calculate new centres
            nd = np.zeros(k)
            for i in range(k):
                nd[i] = np.sum(j == i)
            md = np.maximum(nd, 1)

            x_new = np.zeros((k, p))
            for i in range(k):
                mask = j == i
                if np.any(mask):
                    x_new[i, :] = np.sum(d[mask, :], axis=0) / md[i]
            x = x_new

            # Handle unused centres
            fx = np.where(nd == 0)[0]
            if len(fx) > 0:
                q = np.where(m != 0)[0]
                if len(q) <= len(fx):
                    x[fx[:len(q)], :] = d[q, :]
                else:
                    ri = np.random.permutation(len(q))
                    x[fx, :] = d[q[ri[:len(fx)]], :]

            gg[ll] = np.sum(m)
            if np.array_equal(x, y):
                gg = gg[:ll + 1] / n
                break
        else:
            gg = gg / n

        g = gg[-1]
        return x, g, j, gg
    else:
        # Just calculate distances
        z = v_disteusq(d, x, 'x')
        j = np.argmin(z, axis=1)
        m = z[np.arange(n), j]
        g_val = np.sum(m) / n
        return g_val, j, j, np.array([])
