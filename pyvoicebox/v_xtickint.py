"""V_XTICKINT - Remove non-integer ticks from x-axis."""

import numpy as np


def v_xtickint(ax=None):
    """Remove non-integer tick marks from the x-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes handle. Default is plt.gca().

    Returns
    -------
    xtick : ndarray
        Array of remaining integer tick positions.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    xtick = ax.get_xticks()
    int_ticks = xtick[np.round(xtick) == xtick]
    ax.set_xticks(int_ticks)
    return int_ticks
