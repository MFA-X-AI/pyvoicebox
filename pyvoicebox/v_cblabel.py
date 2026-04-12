"""V_CBLABEL - Add a label to a colorbar."""


def v_cblabel(label, h=None, ax=None):
    """Add a label to a colorbar.

    Parameters
    ----------
    label : str
        Label string for the colorbar.
    h : matplotlib colorbar, axes, or figure, optional
        Handle of the colorbar, axis, or figure.
        If None, searches the current figure for a colorbar.
    ax : matplotlib.axes.Axes, optional
        The axes to search for a nearby colorbar (alternative to h).

    Returns
    -------
    cb : matplotlib colorbar or axes
        Handle of the colorbar that was labelled.
    """
    from pyvoicebox._compat import _require_matplotlib
    plt = _require_matplotlib("v_cblabel")
    import numpy as np

    if h is None and ax is None:
        fig = plt.gcf()
    elif h is not None:
        fig = h
    else:
        fig = ax

    # Helper to find centre of axes position
    def _centre(pos):
        return np.array([pos.x0 + pos.width / 2, pos.y0 + pos.height / 2])

    # If given a colorbar axes directly
    if hasattr(fig, 'get_position') and hasattr(fig, 'colorbar'):
        # This is already a colorbar mappable axes -- just label it
        fig.set_label(label)
        return fig

    # If it's an Axes object
    if hasattr(fig, 'get_position') and hasattr(fig, 'get_lines'):
        # It's an axes -- find colorbars on its parent figure
        target_centre = _centre(fig.get_position())
        parent = fig.get_figure()
        if parent is None:
            raise ValueError('Cannot find parent figure')

        # Find all colorbar axes
        cb_axes = [a for a in parent.get_axes()
                   if hasattr(a, '_colorbar_info') or
                   (hasattr(a, 'get_label') and a.get_label() == '<colorbar>')]
        if not cb_axes:
            raise ValueError('There is no colour bar on this figure')

        if len(cb_axes) == 1:
            cb = cb_axes[0]
        else:
            # Find nearest colorbar
            best = None
            best_dist = float('inf')
            for cba in cb_axes:
                c = _centre(cba.get_position())
                d = np.sum((c - target_centre) ** 2)
                if d < best_dist:
                    best_dist = d
                    best = cba
            cb = best

        cb.set_ylabel(label)
        return cb

    # If it's a Figure object
    if hasattr(fig, 'get_axes'):
        cb_axes = [a for a in fig.get_axes()
                   if hasattr(a, '_colorbar_info') or
                   (hasattr(a, 'get_label') and a.get_label() == '<colorbar>')]
        if not cb_axes:
            raise ValueError('There is no colour bar on this figure')

        if len(cb_axes) == 1:
            cb = cb_axes[0]
        else:
            # Find nearest to current axes
            try:
                cur_ax = plt.gca()
                target_centre = _centre(cur_ax.get_position())
            except Exception:
                target_centre = np.array([0.5, 0.5])

            best = None
            best_dist = float('inf')
            for cba in cb_axes:
                c = _centre(cba.get_position())
                d = np.sum((c - target_centre) ** 2)
                if d < best_dist:
                    best_dist = d
                    best = cba
            cb = best

        cb.set_ylabel(label)
        return cb

    raise ValueError(f'h argument must be colorbar, axis or figure handle')
