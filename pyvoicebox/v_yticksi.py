"""V_YTICKSI - Label the y-axis of a plot using SI multipliers."""

from pyvoicebox.v_xyzticksi import v_xyzticksi


def v_yticksi(ax=None, return_prefix=False):
    """Label the y-axis of a plot using SI multipliers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes handle. Default is plt.gca().
    return_prefix : bool, optional
        If True, return a global SI prefix string that can be
        incorporated into the axis label.

    Returns
    -------
    prefix : str
        The global SI prefix (only if return_prefix=True).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> plt.plot([0, 1, 2], [0, 1000, 2000])
    >>> v_yticksi()
    """
    return v_xyzticksi(2, ax=ax, return_prefix=return_prefix)
