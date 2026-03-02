"""V_FIG2EMF - Save a figure in various image formats."""

import os
import inspect


# Mapping from MATLAB print format strings to matplotlib format strings
_FORMAT_MAP = {
    'meta': 'svg',  # EMF not supported natively, use SVG as substitute
    'svg': 'svg',
    'pdf': 'pdf',
    'eps': 'eps',
    'epsc': 'eps',
    'eps2': 'eps',
    'epsc2': 'eps',
    'ps': 'ps',
    'psc': 'ps',
    'ps2': 'ps',
    'psc2': 'ps',
    'jpeg': 'jpg',
    'png': 'png',
    'tiff': 'tiff',
    'tiffn': 'tiff',
}


def v_fig2emf(h=None, s=None, p=None, f='svg', fig=None):
    """Save a matplotlib figure in various image formats.

    Parameters
    ----------
    h : matplotlib.figure.Figure or str, optional
        Figure handle, or file path string.
        If None, uses the current figure.
    s : str, optional
        File name. Can include '<m>' for calling function name
        and '<n>' for figure number. Default is '<m>_<n>'.
    p : array_like, optional
        If provided, call v_figbolden(p) before saving.
    f : str, optional
        Output format. One of: 'svg', 'pdf', 'eps', 'ps', 'png', 'jpeg',
        'tiff', 'meta' (saved as SVG). Default is 'svg'.
    fig : matplotlib.figure.Figure, optional
        Alternative figure handle.

    Notes
    -----
    The MATLAB 'meta' (EMF) format is not natively supported by matplotlib.
    We use SVG as a substitute which is also a vector format.
    """
    import matplotlib.pyplot as plt

    # Handle flexible argument parsing
    if isinstance(h, str):
        s = h
        h = None

    if fig is not None:
        figure = fig
    elif h is not None:
        figure = h
    else:
        figure = plt.gcf()

    if s is None or s == '':
        s = '<m>_<n>'
    elif s.endswith('/') or s.endswith('\\'):
        s = s + '<m>_<n>'

    # Replace <m> with calling function name
    stack = inspect.stack()
    if len(stack) > 1:
        mfn = stack[-1].function
        if mfn == '<module>':
            frame_file = stack[-1].filename
            mfn = os.path.splitext(os.path.basename(frame_file))[0]
    else:
        mfn = 'Figure'
    s = s.replace('<m>', mfn)

    # Replace <n> with figure number
    fn = str(figure.number)
    s = s.replace('<n>', fn)

    if s == '.':
        return  # suppress save

    # Apply figbolden if requested
    if p is not None:
        from pyvoicebox.v_figbolden import v_figbolden
        if isinstance(p, (int, float)) and p == 0:
            v_figbolden(fig=figure)
        else:
            v_figbolden(pos=p, fig=figure)

    figure.set_tight_layout(True)

    # Map the format
    fmt = _FORMAT_MAP.get(f, f)

    # Determine file extension
    ext_map = {'svg': '.svg', 'pdf': '.pdf', 'eps': '.eps', 'ps': '.ps',
               'png': '.png', 'jpg': '.jpg', 'jpeg': '.jpg', 'tiff': '.tiff'}
    ext = ext_map.get(fmt, '.' + fmt)

    figure.savefig(s + ext, format=fmt, bbox_inches='tight')
