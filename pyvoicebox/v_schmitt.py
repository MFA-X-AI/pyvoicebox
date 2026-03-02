"""V_SCHMITT - Pass input signal through a Schmitt trigger."""

import numpy as np


def v_schmitt(x, thresh=0.5, minwid=0, return_transitions=False):
    """Pass input signal through a Schmitt trigger.

    Parameters
    ----------
    x : array_like
        Input signal.
    thresh : float or array_like, optional
        If scalar in [0,1]: hysteresis fraction. Thresholds are computed from
        max/min of x. If 2-element: [low, high] thresholds. Default is 0.5.
    minwid : int, optional
        Minimum pulse width in samples. Default is 0.
    return_transitions : bool, optional
        If True, return (y_transitions, t) where y_transitions contains
        alternating +1/-1 and t contains transition sample indices.
        If False (default), return full y signal.

    Returns
    -------
    y : ndarray
        If return_transitions=False: signal of same length as x with
        values -1, 0, or +1.
        If return_transitions=True: alternating +1/-1 transition values.
    t : ndarray (only if return_transitions=True)
        Sample indices where x crossed the thresholds (1-based).
    """
    x = np.asarray(x, dtype=float).ravel()
    thresh = np.asarray(thresh, dtype=float).ravel()

    if len(thresh) < 2:
        xmax = np.max(x)
        xmin = np.min(x)
        low_offset = (xmax - xmin) * (1 - thresh[0]) / 2.0
        high = xmax - low_offset
        low = xmin + low_offset
    else:
        low = thresh[0]
        high = thresh[1]

    c = (x > high).astype(int) - (x < low).astype(int)
    # Zero out entries where the value equals the previous
    c[1:] = c[1:] * (c[1:] != c[:-1]).astype(int)

    t = np.where(c != 0)[0]
    if len(t) < 2:
        if return_transitions:
            if len(t) == 0:
                return np.array([], dtype=int), np.array([], dtype=int)
            return c[t], t + 1  # 1-based
        y = np.zeros(len(x), dtype=int)
        if len(t) > 0:
            y[t[0]:] = c[t[0]]
        return y

    # Remove duplicates (consecutive same-sign transitions)
    to_remove = []
    i = 0
    while i < len(t) - 1:
        if c[t[i + 1]] == c[t[i]]:
            to_remove.append(i + 1)
        i += 1
    t = np.delete(t, to_remove)

    # Apply minimum width constraint
    if minwid >= 1:
        # Remove transitions that are too close together
        to_remove = np.where(np.diff(t) < minwid)[0]
        t = np.delete(t, to_remove)
        # Remove duplicates again
        to_remove2 = []
        for i in range(len(t) - 1):
            if c[t[i + 1]] == c[t[i]]:
                to_remove2.append(i + 1)
        if to_remove2:
            t = np.delete(t, to_remove2)

    if return_transitions:
        return c[t], t + 1  # 1-based

    y = np.zeros(len(x), dtype=int)
    if len(t) > 0:
        y_cumul = np.zeros(len(x), dtype=int)
        y_cumul[t] = 2 * c[t]
        y_cumul[t[0]] = c[t[0]]
        y = np.cumsum(y_cumul)

    return y
