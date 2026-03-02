"""V_TILEFIGS - Tile current figure windows."""

import numpy as np


def v_tilefigs(pos=None):
    """Tile current matplotlib figure windows on screen.

    Parameters
    ----------
    pos : array_like, optional
        Virtual screen region [xmin, ymin, width, height].
        Values >= 1 are pixels, values in (0,1) are normalized.
        Default uses the full screen minus taskbar.

    Notes
    -----
    This function works with matplotlib backends that support
    window management (e.g., TkAgg, Qt5Agg). It will have no
    effect with non-interactive backends (e.g., Agg).
    """
    import matplotlib.pyplot as plt

    # Get all open figure numbers sorted
    fig_nums = sorted(plt.get_fignums())
    nf = len(fig_nums)
    if nf == 0:
        return

    # Try to get screen size
    try:
        # For backends with window managers
        root = plt.get_current_fig_manager()
        # Default screen size guess
        scr_w, scr_h = 1920, 1080
    except Exception:
        scr_w, scr_h = 1920, 1080

    if pos is not None:
        pos = np.atleast_1d(np.asarray(pos, dtype=float))
        if np.all(pos < 2):
            # Normalized units
            pos = np.array([
                int(pos[0] * scr_w), int(pos[1] * scr_h),
                int(pos[2] * scr_w), int(pos[3] * scr_h)
            ])
        scr_x, scr_y = int(pos[0]), int(pos[1])
        scr_w, scr_h = int(pos[2]), int(pos[3])
    else:
        scr_x, scr_y = 0, 35  # leave space for taskbar
        scr_h -= 35

    # Find best grid layout (closest to 4:3 aspect ratio)
    best_cols = 1
    best_asp_diff = float('inf')
    for cols in range(1, nf + 1):
        rows = int(np.ceil(nf / cols))
        w = scr_w / cols
        h = scr_h / rows
        asp = w / max(h, 1)
        diff = abs(asp - 4 / 3)
        if diff < best_asp_diff:
            best_asp_diff = diff
            best_cols = cols

    nfh = best_cols
    nfv = int(np.ceil(nf / nfh))
    nfh = int(np.ceil(nf / nfv))

    hpix = scr_w // nfh
    vpix = scr_h // nfv

    for i, fignum in enumerate(fig_nums):
        fig = plt.figure(fignum)
        row = i // nfh
        col = i % nfh
        x = scr_x + col * hpix
        y = scr_y + row * vpix

        try:
            mgr = fig.canvas.manager
            if hasattr(mgr, 'window'):
                # TkAgg backend
                mgr.window.geometry(f'{hpix}x{vpix}+{x}+{y}')
            elif hasattr(mgr, 'resize'):
                mgr.resize(hpix, vpix)
        except Exception:
            # Non-interactive backend -- just set figure size
            dpi = fig.dpi
            fig.set_size_inches(hpix / dpi, vpix / dpi)
