"""V_QUADPEAK - Find quadratically-interpolated peak in an N-D array."""

from __future__ import annotations
import numpy as np


def v_quadpeak(z) -> tuple[float, np.ndarray, int, np.ndarray]:
    """Find quadratically-interpolated peak in an N-D array.

    Parameters
    ----------
    z : array_like
        Input array (at least 3 elements in each non-singleton dimension).

    Returns
    -------
    v : float
        Peak value.
    x : ndarray
        Position of peak (fractional subscript values, 1-based).
    t : int
        -1 for maximum, 0 for saddle point, +1 for minimum.
    m : ndarray
        Matrix defining the fitted quadratic.
    """
    z = np.asarray(z, dtype=float)
    sz_orig = z.shape
    dz_orig = len(sz_orig)

    # In MATLAB, a row vector [1 3 5 4 2] has size [1, 5].
    # In Python, it has shape (5,). Treat 1D arrays as row vectors.
    if dz_orig == 1:
        sz = (1, sz_orig[0])
    else:
        sz = sz_orig
    dz = len(sz)
    psz = z.size

    # Find non-singleton dimensions (0-based)
    mz = [i for i in range(dz) if sz[i] > 1]
    nm = len(mz)
    vz = [sz[i] for i in mz]
    dx = max(mz) + 1 if mz else 1

    if nm == 0:
        raise ValueError('Cannot find peak of a scalar')
    if min(vz) < 3:
        raise ValueError('Need at least 3 points in each non-singleton dimension')

    nc = (nm + 1) * (nm + 2) // 2

    # Build the A matrix using column-major (Fortran) index decomposition
    # to match MATLAB's linear indexing
    a_mat = np.ones((psz, nc))
    ix = np.arange(psz, dtype=float)

    for i in range(nm):
        i1 = i + 1  # MATLAB 1-based loop variable
        dim_size = sz[mz[i]]
        jx = np.floor(ix / dim_size)

        # Linear term column (0-based)
        lin_col = i + nc - nm - 1
        a_mat[:, lin_col] = 1 + ix - jx * dim_size
        ix = jx

        # Quadratic/cross term columns (0-based)
        col_start = (i1 * i1 - i1 + 2) // 2 - 1
        col_end = i1 * (i1 + 1) // 2 - 1  # inclusive

        # Linear terms from dimension 0..i (columns nc-nm-1 to lin_col inclusive)
        lin_range_start = nc - nm - 1
        lin_terms = a_mat[:, lin_range_start:lin_col + 1]  # shape (psz, i1)
        current_lin = a_mat[:, lin_col]  # shape (psz,)

        a_mat[:, col_start:col_end + 1] = lin_terms * current_lin[:, np.newaxis]

    # Solve for polynomial coefficients
    a_mat = np.linalg.solve(a_mat.T @ a_mat, a_mat.T)

    # Use Fortran-order ravel to match MATLAB's column-major linear indexing
    if dz_orig == 1:
        z_flat = z.ravel()
    else:
        z_flat = z.ravel(order='F')

    c = a_mat @ z_flat
    w = np.zeros((nm + 1, nm + 1))

    # Fill w matrix using MATLAB's column-major linear indexing
    idx = np.arange(1, nc + 1)
    j_vals = np.floor((np.sqrt(8 * idx - 7) - 1) / 2).astype(int)
    for k in range(nc):
        lin_idx = int(idx[k] + j_vals[k] * (2 * nm + 1 - j_vals[k]) / 2) - 1
        # MATLAB column-major indexing into (nm+1) x (nm+1) matrix
        row = lin_idx % (nm + 1)
        col = lin_idx // (nm + 1)
        w[row, col] = c[k]
    w = (w + w.T) / 2.0

    mr = w[:nm, :nm]
    we = w[:nm, nm]
    y = -np.linalg.solve(mr, we)
    v = float(y @ we + w[nm, nm])

    # Insert singleton dimensions
    x = np.zeros(dx)
    for i, mi in enumerate(mz):
        x[mi] = y[i]

    m_out = np.zeros((dx + 1, dx + 1))
    mz_ext = list(mz) + [dx]
    for i, mi in enumerate(mz_ext):
        for j_idx, mj in enumerate(mz_ext):
            m_out[mi, mj] = w[i, j_idx]

    ev = np.linalg.eigvalsh(mr)
    t = int(np.all(ev > 0)) - int(np.all(ev < 0))

    return v, x, t, m_out
