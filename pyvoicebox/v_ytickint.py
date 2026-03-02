"""V_YTICKINT - Remove non-integer ticks from y-axis."""

import numpy as np


def v_ytickint(ax=None):
    """Remove non-integer tick marks from the y-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes handle. Default is plt.gca().

    Returns
    -------
    ytick : ndarray
        Array of remaining integer tick positions.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    ytick = ax.get_yticks()
    int_ticks = ytick[np.round(ytick) == ytick]
    ax.set_yticks(int_ticks)
    return int_ticks
