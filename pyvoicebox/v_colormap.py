"""V_COLORMAP - Set and create custom color maps.

Provides custom colormaps and luminance linearization for matplotlib.
"""

from __future__ import annotations
import numpy as np

# CIELUV constants
_lk = (6 / 29) ** 3
_la = 841 / 108
_lb = 4 / 29
_lc = 1.16
_lq = _la * _lc * _lk
_lci = 1 / _lc
_laci = _lci / _la

# Mapping from RGB to luminance (Y from CIE XYZ)
_yv = np.array([0.2126, 0.7152, 0.0722])

# Named custom colormaps: {name: (key_colors, mode, nszs, nmap, power)}
# nmap is the final map size (matching MATLAB nmap=[64 64 64 64])
_CUSTOM_MAPS = {
    'v_thermliny': {
        'colors': np.array([
            [0, 0, 0],
            [72, 0, 167],
            [252, 83, 16],
            [255, 249, 0],
            [255, 255, 255],
        ]) / 255.0,
        'mode': 'y',
        'nszs': [64],
        'nmap': 64,
        'power': 1,
    },
    'v_bipliny': {
        'colors': np.array([
            [0, 0, 0],
            [0, 2, 203],
            [1, 101, 226],
            [128, 128, 128],
            [252, 153, 12],
            [252, 245, 0],
            [252, 249, 18],
            [252, 252, 252],
        ]) / 252.0,
        'mode': 'y',
        'nszs': [64],
        'nmap': 64,
        'power': 1,
    },
    'v_bipveey': {
        'colors': np.array([
            [0, 0.95, 1],
            [0, 0, 0.9],
            [0, 0, 0],
            [0.5, 0, 0],
            [0.80, 0.78, 0],
        ]),
        'mode': 'y',
        'nszs': [33, 31],
        'nmap': 64,
        'power': 1,
    },
    'v_circrby': {
        'colors': np.array([
            [0, 0, 0],
            [1, 0.183, 0],
            [1, 0.9, 0],
            [1, 1, 1],
            [0, 1, 0.8],
            [0, 0.379, 1],
            [0, 0, 0],
        ]),
        'mode': 'y',
        'nszs': [33, 32],
        'nmap': 64,
        'power': 1,
    },
}

# Cache for computed maps
_computed_maps = {}


def _luminance_to_lightness(y):
    """Convert luminance to CIE lightness L*."""
    return _lc * (_la * y + (y > _lk) * (y ** (1 / 3) - _la * y - _lb))


def _lightness_to_luminance(l_star):
    """Convert CIE lightness L* to luminance."""
    return _laci * l_star + (l_star > _lq) * ((_lci * l_star + _lb) ** 3 - _laci * l_star)


def v_colormap(map_input=None, m='', n=None, p=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create or modify a color map.

    Parameters
    ----------
    map_input : str or ndarray, optional
        Either an (r, 3) RGB array or a string naming a colormap.
        Custom maps: 'v_thermliny', 'v_bipliny', 'v_bipveey', 'v_circrby'.
        Standard matplotlib maps can also be specified by name.
        If None, returns the default 'viridis' colormap.
    m : str, optional
        Mode string:
        'y' - force luminance^p to be linear or V-shaped
        'l' - force lightness^p to be linear or V-shaped
        'Y' - like 'y' but with default p=2/3
        'L' - like 'l' but with default p=2
        'f' - flip the map
        'b'/'B' - force minimum luminance >= 0.05/0.10
        'w'/'W' - force maximum luminance <= 0.95/0.90
        'g' - plot information about the colormap
        'k' - keep the current colormap
    n : int or array_like, optional
        Number of entries in the colormap, or number per segment.
    p : float, optional
        Power law exponent for luminance/lightness linearization.

    Returns
    -------
    rgb : ndarray
        (N, 3) RGB colormap array with values in [0, 1].
    y : ndarray
        Column vector of luminance values.
    l : ndarray
        Column vector of lightness values.
    """
    global _computed_maps

    # Handle power law defaults
    if p is None:
        if 'Y' in m:
            p = 2 / 3
        elif 'L' in m:
            p = 2
        else:
            p = 1
    pr = 1 / p

    um = m  # preserve original case
    m_lower = m.lower()

    # Get the base RGB map
    if map_input is None:
        # Default: use a standard 64-entry viridis-like map
        from pyvoicebox._compat import _require_matplotlib
        plt = _require_matplotlib("v_colormap")
        cmap = plt.cm.get_cmap('viridis', 64)
        rgb = cmap(np.linspace(0, 1, 64))[:, :3]
    elif isinstance(map_input, str):
        name_lower = map_input.lower()
        # Check custom maps
        matched_name = None
        for k in _CUSTOM_MAPS:
            if k.lower() == name_lower:
                matched_name = k
                break

        if matched_name is not None:
            if matched_name in _computed_maps:
                rgb = _computed_maps[matched_name].copy()
            else:
                spec = _CUSTOM_MAPS[matched_name]
                rgb = v_colormap(spec['colors'], spec['mode'], spec['nszs'], spec['power'])[0]
                nmap = spec['nmap']
                rgb = rgb[:nmap, :]
                _computed_maps[matched_name] = rgb.copy()
        else:
            # Try matplotlib built-in colormap
            from pyvoicebox._compat import _require_matplotlib
            plt = _require_matplotlib("v_colormap")
            try:
                cmap = plt.cm.get_cmap(map_input, 64)
                rgb = cmap(np.linspace(0, 1, 64))[:, :3]
            except ValueError:
                raise ValueError(f'Unknown colormap: {map_input}')
    else:
        rgb = np.asarray(map_input, dtype=float).copy()
        if rgb.ndim == 1:
            rgb = rgb.reshape(-1, 3)

    # Linear interpolation and/or luminance linearization
    if ('y' in m_lower or 'l' in m_lower) or (n is not None):
        nm = rgb.shape[0]

        if 'y' in m_lower or 'l' in m_lower:
            y_lum = rgb @ _yv  # luminance

            # Find monotonic segments
            up = y_lum[1:] > y_lum[:-1]
            if nm > 2:
                ex = up[:-1] != up[1:]
                n_extrema = int(np.sum(ex))
            else:
                n_extrema = 0

            if n_extrema == 0:
                # Monotonic
                if n is None:
                    r = nm
                else:
                    n_arr = np.atleast_1d(np.asarray(n, dtype=int)).ravel()
                    r = n_arr[0]

                if 'y' in m_lower:
                    l_vals = y_lum[np.array([0, nm - 1])] ** p
                    tt = (l_vals[0] + np.arange(r) * (l_vals[1] - l_vals[0]) / (r - 1)) ** pr
                else:
                    tt_y = y_lum[np.array([0, nm - 1])]
                    l_vals = _luminance_to_lightness(tt_y) ** p
                    tt_l = (l_vals[0] + np.arange(r) * (l_vals[1] - l_vals[0]) / (r - 1)) ** pr
                    tt = _lightness_to_luminance(tt_l)

                yd = 1 if (y_lum[1] > y_lum[0]) else -1
                combined = np.concatenate([tt, y_lum])
                ix = np.argsort(combined * yd)

            elif n_extrema == 1:
                # V-shaped
                ipk = int(np.where(ex)[0][0]) + 1

                if n is None:
                    n_segs = [ipk, nm - ipk]
                else:
                    n_arr = np.atleast_1d(np.asarray(n, dtype=int)).ravel()
                    n_segs = [int(n_arr[0]), int(n_arr[1])]

                r = n_segs[0] + n_segs[1]

                if 'y' in m_lower:
                    l_vals = y_lum[np.array([0, ipk, nm - 1])] ** p
                    seg1 = l_vals[1] + np.arange(1 - n_segs[0], 1) * (l_vals[1] - l_vals[0]) / (n_segs[0] - 1)
                    seg2 = l_vals[1] + np.arange(1, n_segs[1] + 1) * (l_vals[2] - l_vals[1]) / n_segs[1]
                    tt = np.concatenate([seg1, seg2]) ** pr
                else:
                    tt_y = y_lum[np.array([0, ipk, nm - 1])]
                    l_vals = _luminance_to_lightness(tt_y) ** p
                    seg1 = l_vals[1] + np.arange(1 - n_segs[0], 1) * (l_vals[1] - l_vals[0]) / (n_segs[0] - 1)
                    seg2 = l_vals[1] + np.arange(1, n_segs[1] + 1) * (l_vals[2] - l_vals[1]) / n_segs[1]
                    tt_l = np.concatenate([seg1, seg2]) ** pr
                    tt = _lightness_to_luminance(tt_l)

                yd = 1 if (y_lum[1] > y_lum[0]) else -1
                y_pk = y_lum[ipk]
                combined = np.concatenate([
                    tt[:n_segs[0]] - y_pk,
                    y_pk - tt[n_segs[0]:r],
                    y_lum[:ipk + 1] - y_pk,
                    y_pk - y_lum[ipk + 1:],
                ])
                ix = np.argsort(combined * yd)

            else:
                raise ValueError('luminance has more than two monotonic segments')

        else:
            # Just linearly interpolate
            n_arr = np.atleast_1d(np.asarray(n, dtype=int)).ravel()
            if len(n_arr) == nm - 1:
                r = int(np.sum(n_arr))
                y_interp = np.concatenate([[1], np.cumsum(n_arr, dtype=float)])
            else:
                r = n_arr[0]
                y_interp = 1 + np.arange(nm) * (r - 1) / (nm - 1)

            tt = np.arange(1, r + 1, dtype=float)
            combined = np.concatenate([tt, y_interp])
            ix = np.argsort(combined)

        # Perform the interpolation
        jx = np.zeros(len(ix), dtype=int)
        jx[ix] = np.arange(len(ix))
        jx = jx[:r]
        jx = jx - np.arange(r)
        jx = np.clip(jx, 1, nm - 1) - 1  # convert to 0-indexed lower bound

        if 'y' in m_lower or 'l' in m_lower:
            al = (tt - y_lum[jx]) / (y_lum[jx + 1] - y_lum[jx] + 1e-300)
        else:
            al = (tt - y_interp[jx]) / (y_interp[jx + 1] - y_interp[jx] + 1e-300)
        al = al[:, np.newaxis]
        rgb = rgb[jx, :] + (rgb[jx + 1, :] - rgb[jx, :]) * al

    # Flip if requested
    if 'f' in m_lower:
        rgb = rgb[::-1, :]

    # Compute luminance
    y_out = rgb @ _yv

    # Constrain luminance if requested
    if 'b' in m_lower or 'w' in m_lower:
        minyt = 0.05 * (('b' in m_lower) + ('B' in um))
        maxyt = 1 - 0.05 * (('w' in m_lower) + ('W' in um))
        maxy = np.max(y_out)
        miny = np.min(y_out)
        if maxy > maxyt or miny < minyt:
            maxy = max(maxy, maxyt)
            miny = min(miny, minyt)
            rgb = (rgb - miny) * (maxyt - minyt) / (maxy - miny) + minyt
            y_out = rgb @ _yv

    # Compute lightness
    l_out = _luminance_to_lightness(y_out)

    # Clip to valid range
    rgb = np.clip(rgb, 0, 1)

    return rgb, y_out, l_out


def v_colormap_to_mpl(name_or_rgb, m='', n=None, p=None):
    """Create a matplotlib LinearSegmentedColormap from a v_colormap specification.

    Parameters
    ----------
    name_or_rgb : str or ndarray
        Colormap name or RGB array (same as v_colormap).
    m : str, optional
        Mode string (same as v_colormap).
    n : int or array_like, optional
        Number of entries (same as v_colormap).
    p : float, optional
        Power law (same as v_colormap).

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        A matplotlib colormap object.
    """
    try:
        from matplotlib.colors import ListedColormap
    except ImportError as e:
        raise ImportError(
            "v_colormap_to_mpl requires matplotlib, which is an optional dependency. "
            "Install it with: pip install 'pyvoicebox[plot]'"
        ) from e

    rgb, _, _ = v_colormap(name_or_rgb, m, n, p)
    if isinstance(name_or_rgb, str):
        cmap_name = name_or_rgb
    else:
        cmap_name = 'v_custom'
    return ListedColormap(rgb, name=cmap_name)
