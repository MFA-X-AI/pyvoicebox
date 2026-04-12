"""V_ENTROPY - Shannon entropy of discrete and sampled continuous distributions."""

from __future__ import annotations
import numpy as np


def v_entropy(p, dim=None, cond=None, arg=None, step=None) -> np.ndarray:
    """Calculate the entropy of discrete and sampled continuous distributions.

    Parameters
    ----------
    p : array_like
        Probability array.
    dim : int or list of int, optional
        Dimensions along which to evaluate entropy. Default: first non-singleton.
    cond : list of int, optional
        Dimensions to use as conditional variables.
    arg : list of int, optional
        Dimensions to use as parameters in output.
    step : float or list of float, optional
        Sample increment for continuous distributions.

    Returns
    -------
    h : ndarray
        Entropy value(s).
    """
    p = np.asarray(p, dtype=float)
    ndim = p.ndim

    # Handle step
    if step is None:
        stp = np.zeros(max(ndim, 1))
    else:
        step = np.atleast_1d(step)
        stp = np.full(max(ndim, 1), step[0])
        stp[:len(step)] = step
        stp = np.log2(stp)

    # Handle arg
    if arg is None:
        arg = []
    else:
        arg = [a for a in np.atleast_1d(arg) if a >= 0]

    # Handle cond
    if cond is None:
        cond = []
    else:
        cond = [c for c in np.atleast_1d(cond) if c >= 0]

    if not cond:
        s = p.shape
        if dim is None:
            # First non-singleton or dimension with size >= 2
            candidates = [i for i, si in enumerate(s) if si >= min(2, max(s))]
            dim = [candidates[0]] if candidates else [0]
        else:
            dim = [d for d in np.atleast_1d(dim) if d >= 0]

        sd = 1
        for d in dim:
            if d < len(s):
                sd *= s[d]

        if sd == 1:
            # Bernoulli variables
            q = p.ravel()
            h = -np.log2(q + (q == 0)) * q - np.log2(1 - q + (q == 1)) * (1 - q)
            return h
        else:
            # General case for 1D/2D
            if p.ndim == 1:
                q = p
                sq = np.sum(q)
                h = np.sum(-np.log2(q + (q == 0)) * q) / sq + np.log2(sq)
            else:
                # Sum along the specified dimensions
                axes = tuple(dim)
                sq = np.sum(p, axis=axes, keepdims=True)
                h = np.sum(-np.log2(p + (p == 0)) * p, axis=axes, keepdims=True) / sq + np.log2(sq)
                h = np.squeeze(h)

            h = h + np.sum(stp[dim])
            return h
    else:
        # Conditional entropy: H(X|Y) = H(X,Y) - H(Y)
        joint_dims = list(dim) + list(cond)
        return v_entropy(p, joint_dims, arg=arg) - v_entropy(p, cond, arg=arg)
