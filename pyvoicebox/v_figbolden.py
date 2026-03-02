"""V_FIGBOLDEN - Embolden, resize and recolour the current figure."""

import numpy as np


def v_figbolden(pos=None, pv=None, m='', fig=None):
    """Embolden, resize and recolour a matplotlib figure.

    Parameters
    ----------
    pos : array_like, optional
        Figure size as [width, height] in pixels, or [xmin, ymin, width, height].
        If a single negative number, fix aspect ratio to -width/height preserving area.
        If a single positive number, use 4:3 aspect ratio.
    pv : dict, optional
        Dictionary of property-value pairs to apply. Default is:
        {'fontname': 'Arial', 'fontsize': 16, 'linewidth': 2, 'markersize': 8}
    m : str, optional
        Mode string:
        'l' - list changes made (print to stdout)
        'd' - use default pv settings
        'c' - change default colours for better contrast
        'x' - suppress all changes
    fig : matplotlib.figure.Figure, optional
        Figure handle. Default is plt.gcf().
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.gcf()

    if pv is None or 'd' in m:
        pv = {'fontname': 'Arial', 'fontsize': 16, 'linewidth': 2, 'markersize': 8}

    do_list = 'l' in m
    do_changes = 'x' not in m

    # Resize figure if pos is specified
    if pos is not None:
        pos = np.atleast_1d(np.asarray(pos, dtype=float)).ravel()
        if len(pos) >= 4:
            fig.set_size_inches(pos[2] / fig.dpi, pos[3] / fig.dpi)
        elif len(pos) >= 2:
            fig.set_size_inches(pos[0] / fig.dpi, pos[1] / fig.dpi)
        elif len(pos) == 1:
            if pos[0] > 0:
                fig.set_size_inches(pos[0] / fig.dpi, 0.75 * pos[0] / fig.dpi)
            else:
                w, h = fig.get_size_inches()
                area = w * h
                ratio = -pos[0]
                new_h = np.sqrt(area / ratio)
                new_w = ratio * new_h
                fig.set_size_inches(new_w, new_h)

    # Change default colours for better contrast
    if 'c' in m:
        color_map = {
            (0, 0.5, 0): (0, 0.7, 0),          # green
            (0, 0.75, 0.75): (0, 0.7, 0.7),     # cyan
            (0.75, 0.75, 0): (0.83, 0.83, 0),    # yellow
        }
        for ax in fig.get_axes():
            for line in ax.get_lines():
                c = line.get_color()
                if isinstance(c, str):
                    continue
                c_tuple = tuple(np.round(c[:3], 2))
                for old_c, new_c in color_map.items():
                    if np.allclose(c_tuple, old_c, atol=0.05):
                        if do_changes:
                            line.set_color(new_c)
                        if do_list:
                            print(f'  change Color: {old_c} -> {new_c}')

    # Apply property-value pairs to all text and line objects
    if do_changes:
        for ax in fig.get_axes():
            # Apply to axes labels and title
            for text_obj in [ax.title, ax.xaxis.label, ax.yaxis.label]:
                if 'fontsize' in pv:
                    text_obj.set_fontsize(pv['fontsize'])
                if 'fontname' in pv:
                    text_obj.set_fontname(pv['fontname'])

            # Apply to tick labels
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                if 'fontsize' in pv:
                    label.set_fontsize(pv['fontsize'])
                if 'fontname' in pv:
                    label.set_fontname(pv['fontname'])

            # Apply to lines
            for line in ax.get_lines():
                if 'linewidth' in pv:
                    line.set_linewidth(pv['linewidth'])
                if 'markersize' in pv:
                    line.set_markersize(pv['markersize'])

            # Apply to legend if present
            legend = ax.get_legend()
            if legend is not None:
                for text_obj in legend.get_texts():
                    if 'fontsize' in pv:
                        text_obj.set_fontsize(pv['fontsize'])
                    if 'fontname' in pv:
                        text_obj.set_fontname(pv['fontname'])
