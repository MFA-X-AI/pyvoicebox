"""V_XYZTICKSI - Label an axis of a plot using SI multipliers."""

import numpy as np


# SI prefix table: index 0 = 'y' (10^-24), index 8 = '' (10^0), index 16 = 'Y' (10^24)
_SI_PREFIXES = ['y', 'z', 'a', 'f', 'p', 'n', '\u03bc', 'm', '', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']


def _format_si_tick(value, dp, si_power, global_si=False):
    """Format a single tick value with SI prefix.

    Parameters
    ----------
    value : float
        The integer mantissa value (before applying decimal point).
    dp : int
        Number of decimal places.
    si_power : int
        SI multiplier power (multiple of 3).
    global_si : bool
        If True, suppress the SI prefix in the label.

    Returns
    -------
    label : str
        Formatted tick label string.
    """
    if dp > 0:
        label = f'{value * 10**(-dp):.{dp}f}'
    else:
        label = f'{int(value)}'

    if not global_si and si_power != 0 and value != 0:
        idx = si_power // 3 + 8
        if 0 <= idx < len(_SI_PREFIXES):
            label += _SI_PREFIXES[idx]

    return label


def v_xyzticksi(ax_idx, ax=None, return_prefix=False):
    """Label an axis of a plot using SI multipliers.

    This is the core implementation called by v_xticksi and v_yticksi.

    Parameters
    ----------
    ax_idx : int
        Axis index: 1 for x-axis, 2 for y-axis, 3 for z-axis.
    ax : matplotlib.axes.Axes, optional
        Axes handle. Default is plt.gca().
    return_prefix : bool, optional
        If True, try to use a global SI prefix and return it.

    Returns
    -------
    prefix : str
        The global SI prefix string (only if return_prefix is True).
        Empty string if no global prefix could be used.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if ax is None:
        ax = plt.gca()

    # Get axis properties
    if ax_idx == 1:
        a, b = ax.get_xlim()
        scale = ax.get_xscale()
        axis_obj = ax.xaxis
    elif ax_idx == 2:
        a, b = ax.get_ylim()
        scale = ax.get_yscale()
        axis_obj = ax.yaxis
    else:
        if hasattr(ax, 'get_zlim'):
            a, b = ax.get_zlim()
            scale = ax.get_zscale() if hasattr(ax, 'get_zscale') else 'linear'
            axis_obj = ax.zaxis
        else:
            return ''

    if a >= b:
        return ''

    # Get current tick positions
    ticks = axis_obj.get_ticklocs()
    ticks = ticks[(ticks >= a) & (ticks <= b)]

    if len(ticks) == 0:
        return ''

    # Determine if we can use a global SI prefix
    max_abs = max(abs(a), abs(b))
    if max_abs == 0:
        return ''

    if return_prefix:
        # Try to find a global SI prefix
        e = int(np.floor(np.log10(max_abs)))
        gi = 3 * (e // 3)
        gi = max(-24, min(24, gi))
        prefix = _SI_PREFIXES[gi // 3 + 8]

        # Scale all ticks
        g = 10.0 ** gi
        scaled_ticks = ticks / g

        # Create labels
        labels = []
        for t in scaled_ticks:
            if t == 0:
                labels.append('0')
            else:
                # Determine appropriate decimal places
                if t == int(t):
                    labels.append(f'{int(t)}')
                else:
                    labels.append(f'{t:g}')

        if ax_idx == 1:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        elif ax_idx == 2:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
        else:
            ax.set_zticks(ticks)
            ax.set_zticklabels(labels)

        return prefix
    else:
        # Format each tick with its own SI prefix
        labels = []
        for t in ticks:
            if t == 0:
                labels.append('0')
            else:
                abs_t = abs(t)
                e = int(np.floor(np.log10(abs_t)))
                si = 3 * (e // 3)
                si = max(-24, min(24, si))
                scaled = t / (10.0 ** si)
                prefix = _SI_PREFIXES[si // 3 + 8]
                if scaled == int(scaled):
                    label = f'{int(scaled)}'
                else:
                    label = f'{scaled:g}'
                if prefix:
                    label += prefix
                labels.append(label)

        if ax_idx == 1:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        elif ax_idx == 2:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
        else:
            ax.set_zticks(ticks)
            ax.set_zticklabels(labels)

        return ''
