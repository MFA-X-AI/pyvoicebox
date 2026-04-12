"""V_TEXTHVC - Write text on graph with specified alignment and colour."""

from __future__ import annotations
import numpy as np


# Colour character mapping
_COLOR_MAP = {
    'r': 'red', 'g': 'green', 'b': 'blue',
    'c': 'cyan', 'm': 'magenta', 'y': 'yellow',
    'k': 'black', 'w': 'white',
}


def v_texthvc(x, y, t, p=None, q=None, r=None, ax=None) -> np.ndarray:
    """Write text on graph with specified alignment and colour.

    Parameters
    ----------
    x : float or array_like
        X-position. If length-2 array [x0, k], positions at x0 + k*axis_width.
    y : float or array_like
        Y-position. If length-2 array [y0, k], positions at y0 + k*axis_height.
    t : str
        Text string to display.
    p : str, optional
        Alignment/colour string 'hvc' where:
        h = horizontal: l=left, c/m=center, r=right
        v = vertical: t=top, p=cap, c/m=middle, l=baseline, b=bottom
        c = colour: r,g,b,c,m,y,k,w
        If h or v is uppercase, position is normalized (0 to 1).
    q : array_like or str, optional
        Alternative colour as [r, g, b] or a colour string.
    r : optional
        Legacy colour argument (for 6-argument compatibility).
    ax : matplotlib.axes.Axes, optional
        Axes handle. Default is plt.gca().

    Returns
    -------
    text_obj : matplotlib.text.Text
        The text object.
    """
    from pyvoicebox._compat import _require_matplotlib
    plt = _require_matplotlib("v_texthvc")
    if ax is None:
        ax = plt.gca()

    ha_map = {'l': 'left', 'c': 'center', 'm': 'center', 'r': 'right'}
    va_map = {'t': 'top', 'p': 'top', 'c': 'center', 'm': 'center', 'l': 'baseline', 'b': 'bottom'}

    kwargs = {}

    if p is None:
        # No alignment or colour specified
        return ax.text(x, y, t)

    if r is not None and q is not None and p is not None:
        # 6-argument legacy: p=ha, q=va, r=color
        kwargs['horizontalalignment'] = p
        kwargs['verticalalignment'] = q
        kwargs['color'] = r
        return ax.text(x, y, t, **kwargs)

    # Determine colour
    if q is not None:
        color = q
    elif len(p) >= 3:
        c_char = p[2:]
        if len(c_char) == 1 and c_char in _COLOR_MAP:
            color = _COLOR_MAP[c_char]
        else:
            color = c_char
    else:
        color = 'black'

    # Horizontal alignment
    h_char = p[0]
    h_lower = h_char.lower()
    if h_lower not in ha_map:
        raise ValueError(f'Invalid horizontal spec: {h_char}')
    kwargs['horizontalalignment'] = ha_map[h_lower]

    # Vertical alignment
    v_char = p[1]
    v_lower = v_char.lower()
    if v_lower not in va_map:
        raise ValueError(f'Invalid vertical spec: {v_char}')
    kwargs['verticalalignment'] = va_map[v_lower]

    kwargs['color'] = color

    # Handle x positioning
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if len(x) > 1:
        xlim = ax.get_xlim()
        if ax.get_xscale() == 'log':
            x_val = np.exp(np.log(x[0]) + np.log(xlim[1] / xlim[0]) * x[1])
        else:
            x_val = x[0] + (xlim[1] - xlim[0]) * x[1]
    else:
        x_val = x[0]
        if h_char == h_char.upper() and h_char.upper() != h_char.lower():
            xlim = ax.get_xlim()
            if ax.get_xscale() == 'log':
                x_val = np.exp(np.log(xlim[0]) * (1 - x_val) + np.log(xlim[1]) * x_val)
            else:
                x_val = xlim[0] * (1 - x_val) + xlim[1] * x_val

    # Handle y positioning
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if len(y) > 1:
        ylim = ax.get_ylim()
        if ax.get_yscale() == 'log':
            y_val = np.exp(np.log(y[0]) + np.log(ylim[1] / ylim[0]) * y[1])
        else:
            y_val = y[0] + (ylim[1] - ylim[0]) * y[1]
    else:
        y_val = y[0]
        if v_char == v_char.upper() and v_char.upper() != v_char.lower():
            ylim = ax.get_ylim()
            if ax.get_yscale() == 'log':
                y_val = np.exp(np.log(ylim[0]) * (1 - y_val) + np.log(ylim[1]) * y_val)
            else:
                y_val = ylim[0] * (1 - y_val) + ylim[1] * y_val

    return ax.text(x_val, y_val, t, **kwargs)
