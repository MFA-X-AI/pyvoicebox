"""V_UPOLYHEDRON - calculate uniform polyhedron characteristics.

This is a simplified implementation that supports the most common polyhedra
by name or index. The full MATLAB implementation supports all 75+ uniform
polyhedra with Wythoff symbol computation.
"""

from __future__ import annotations
import numpy as np


# Precomputed vertex lists for common polyhedra
def _tetrahedron():
    """Regular tetrahedron (index 1)."""
    v = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float) / np.sqrt(3)
    return v


def _octahedron():
    """Regular octahedron (index 2)."""
    v = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=float)
    return v


def _cube():
    """Regular cube/hexahedron (index 6)."""
    s = 1.0 / np.sqrt(3)
    v = np.array([
        [-s, -s, -s], [-s, -s, s], [-s, s, -s], [-s, s, s],
        [s, -s, -s], [s, -s, s], [s, s, -s], [s, s, s],
    ])
    return v


def _icosahedron():
    """Regular icosahedron (index 3)."""
    phi = (1 + np.sqrt(5)) / 2
    r = np.sqrt(1 + phi ** 2)
    v = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ]) / r
    return v


def _dodecahedron():
    """Regular dodecahedron (index 4)."""
    phi = (1 + np.sqrt(5)) / 2
    r = np.sqrt(3)
    v = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                v.append([s1, s2, s3])
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            v.append([0, s1 * phi, s2 / phi])
            v.append([s1 / phi, 0, s2 * phi])
            v.append([s1 * phi, s2 / phi, 0])
    v = np.array(v) / r
    return v


_POLYHEDRA = {
    'tetrahedron': _tetrahedron,
    'octahedron': _octahedron,
    'icosahedron': _icosahedron,
    'dodecahedron': _dodecahedron,
    'cube': _cube,
    'hexahedron': _cube,
}

_INDEX_MAP = {
    1: 'tetrahedron',
    2: 'octahedron',
    3: 'icosahedron',
    4: 'dodecahedron',
    6: 'cube',
}


def v_upolyhedron(w, md='') -> np.ndarray:
    """Calculate uniform polyhedron characteristics.

    This is a simplified implementation supporting common polyhedra.

    Parameters
    ----------
    w : int or str
        Polyhedron specification (index or name).
    md : str, optional
        Mode string.

    Returns
    -------
    vlist : ndarray
        Vertex list with columns [x, y, z, d, n, e, t].
    """
    if isinstance(w, str):
        name = w.lower()
        if name not in _POLYHEDRA:
            raise ValueError(f'Unknown polyhedron: {w}')
        vertices = _POLYHEDRA[name]()
    elif isinstance(w, (int, float)):
        idx = int(w)
        if idx not in _INDEX_MAP:
            raise ValueError(f'Polyhedron index {idx} not supported in simplified implementation')
        vertices = _POLYHEDRA[_INDEX_MAP[idx]]()
    else:
        raise ValueError('w must be a string name or integer index')

    nv = vertices.shape[0]
    d = np.sqrt(np.sum(vertices ** 2, axis=1))
    vlist = np.column_stack([
        vertices,
        d,
        np.zeros(nv),  # valency placeholder
        np.zeros(nv),  # edge index placeholder
        np.ones(nv),   # type placeholder
    ])

    return vlist
