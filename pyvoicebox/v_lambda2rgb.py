"""V_LAMBDA2RGB - Convert wavelength to XYZ or RGB colour space."""

import numpy as np

# Coefficients for 1931 standard observer
_c = np.array([
    1.065, -0.5 / 33.33**2, 595.8,
    0.366, -0.5 / 19.44**2, 446.8,
    1.014, -0.5 / 0.075**2, np.log(556.3),
    1.839, -0.5 / 0.051**2, np.log(449.8),
])

# Coefficients for 1964 standard observer
_d = np.array([
    0.398, -1250, -570.1, -np.log(1014),
    1.132, -234, 1338, 1300, -np.log(743.5),
    1.011, -0.5 / 46.14**2, 556.1,
    2.06, -32, 265.8, -np.log(180.4),
])

# CIE XYZ to RGB matrix
# In MATLAB: xr = [0.49 0.31 0.2; 0.17697 0.8124 0.01063; 0 0.01 0.99];
#            xr = xr'/xr(2);   % xr(2) is column-major index 2 = element (2,1) = 0.17697
#            rx = inv(xr);
_xr_orig = np.array([
    [0.49, 0.31, 0.2],
    [0.17697, 0.8124, 0.01063],
    [0, 0.01, 0.99],
])
# xr(2) in MATLAB = second element in column-major = (row 1, col 0) = 0.17697
_xr_mat = _xr_orig.T / _xr_orig[1, 0]
_rx_mat = np.linalg.inv(_xr_mat)


def v_lambda2rgb(l, m='r'):
    """Convert wavelength to XYZ or RGB colour space.

    Parameters
    ----------
    l : array_like
        Wavelength(s) in nanometres.
    m : str, optional
        Mode:
        'r' - output is [R G B] using 1931 observer with negatives clipped (default)
        's' - output is [R G B] using 1931 observer with signed values
        'x' - output is [X Y Z] using 1931 observer
        Use uppercase 'R', 'S', 'X' for 1964 observer.

    Returns
    -------
    x : ndarray
        Tristimulus values, shape (n, 3).
    """
    l = np.atleast_1d(np.asarray(l, dtype=float))
    lv = l.ravel()
    lm = m.lower()

    if m == lm:  # lowercase = 1931 standard observer
        ll = np.log(lv)
        x1 = _c[0] * np.exp(_c[1] * (lv - _c[2])**2) + _c[3] * np.exp(_c[4] * (lv - _c[5])**2)
        x2 = _c[6] * np.exp(_c[7] * (ll - _c[8])**2)
        x3 = _c[9] * np.exp(_c[10] * (ll - _c[11])**2)
        x = np.column_stack([x1, x2, x3])
    else:  # uppercase = 1964 standard observer
        x1 = (_d[0] * np.exp(_d[1] * (np.log(lv - _d[2]) + _d[3])**2) +
              _d[4] * np.exp(_d[5] * (np.log(_d[6] - np.minimum(lv, _d[7])) + _d[8])**2))
        x2 = _d[9] * np.exp(_d[10] * (lv - _d[11])**2)
        x3 = _d[12] * np.exp(_d[13] * (np.log(lv - _d[14]) + _d[15])**2)
        x = np.column_stack([x1, x2, x3])

    if lm == 's':
        x = x @ _rx_mat
    elif lm == 'r':
        x = np.maximum(x @ _rx_mat, 0)
    # else 'x' mode: return XYZ directly

    return x
