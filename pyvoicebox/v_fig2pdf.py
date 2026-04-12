"""V_FIG2PDF - Save a figure to PDF/EPS/PS format."""

from __future__ import annotations
import os
import inspect


def v_fig2pdf(h=None, s=None, p=None, f='p', fig=None) -> None:
    """Save a matplotlib figure to PDF, EPS or PS format.

    Parameters
    ----------
    h : matplotlib.figure.Figure or str, optional
        Figure handle, or file path string (for convenience).
        If None, uses the current figure.
    s : str, optional
        File name. Can include '<m>' for the calling function name
        and '<n>' for the figure number. If ending with '/' or '\\',
        '<m>_<n>' is appended. Default is '<m>_<n>'.
    p : array_like, optional
        If provided, call v_figbolden(p) before saving.
    f : str, optional
        Output format string: 'p' for PDF (default), 'e' for EPS, 's' for PS.
    fig : matplotlib.figure.Figure, optional
        Alternative figure handle (overrides h if both provided).

    Notes
    -----
    Unlike the MATLAB version, this does not require MikTeX/pdfcrop.
    Uses matplotlib's built-in savefig with tight_layout.
    """
    from pyvoicebox._compat import _require_matplotlib
    plt = _require_matplotlib("v_fig2pdf")
    # Handle flexible argument parsing like MATLAB version
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

    figure.set_layout_engine('tight')

    # Save in requested formats
    if not f:
        f = 'p'

    if 'p' in f:
        figure.savefig(s + '.pdf', format='pdf', bbox_inches='tight')
    if 'e' in f:
        figure.savefig(s + '.eps', format='eps', bbox_inches='tight')
    if 's' in f:
        figure.savefig(s + '.ps', format='ps', bbox_inches='tight')
