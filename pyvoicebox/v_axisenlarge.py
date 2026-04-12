"""V_AXISENLARGE - Enlarge the axes of a figure."""

import numpy as np


def v_axisenlarge(f=None, ax=None):
    """Enlarge the axes of a matplotlib plot.

    Parameters
    ----------
    f : float or array_like, optional
        Enlarge axis by a factor f relative to current size.
        If negative, shrink to fit content before enlarging by abs(f).
        Can be scalar, [fx, fy], [fx, fy, fz],
        [fleft, fright, fbottom, ftop], or [fleft, fright, fbottom, ftop, fdown, fup].
        Default is -1.02.
    ax : matplotlib.axes.Axes, optional
        Axes handle. Default is current axes (plt.gca()).
    """
    from pyvoicebox._compat import _require_matplotlib
    plt = _require_matplotlib("v_axisenlarge")
    if ax is None:
        ax = plt.gca()
    if f is None:
        f = -1.02

    f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()

    # Expansion table (1-indexed in MATLAB, 0-indexed here)
    fpt = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 2, 2],
        [0, 1, 2, 3, 2, 3],
        [0, 1, 2, 3, 4, 4],
        [0, 1, 2, 3, 4, 5],
    ])
    nf = min(len(f), 6)
    f = f[fpt[nf - 1, :]]  # expand f to 6 elements

    # Get current limits
    xlim = list(ax.get_xlim())
    ylim = list(ax.get_ylim())
    # Check if 3D
    is_3d = hasattr(ax, 'get_zlim')
    if is_3d:
        zlim = list(ax.get_zlim())
    else:
        zlim = [0.0, 1.0]

    ax0 = np.array([xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1]])

    if np.any(f >= 0):
        # Keep current limits where f >= 0
        pass
    else:
        ax0 = np.zeros(6)

    if np.any(f < 0):
        # Auto-fit to tight limits
        ax.autoscale(enable=True, tight=True)
        ax.autoscale_view(tight=True)
        xlim_t = ax.get_xlim()
        ylim_t = ax.get_ylim()
        if is_3d:
            zlim_t = ax.get_zlim()
        else:
            zlim_t = (0.0, 1.0)
        ax1 = np.array([xlim_t[0], xlim_t[1], ylim_t[0], ylim_t[1], zlim_t[0], zlim_t[1]])
        mask = f < 0
        ax0[mask] = ax1[mask]
        f = np.abs(f)

    # ax1 = ax0 * f + ax0([1,0,3,2,5,4]) * (1 - f)
    ax0_swap = ax0[np.array([1, 0, 3, 2, 5, 4])]
    ax1 = ax0 * f + ax0_swap * (1 - f)

    ax.set_xlim(ax1[0], ax1[1])
    ax.set_ylim(ax1[2], ax1[3])
    if is_3d:
        ax.set_zlim(ax1[4], ax1[5])
