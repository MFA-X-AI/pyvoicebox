"""V_RECTIFYHOMOG - apply rectifying homographies to an image set."""

import numpy as np
from pyvoicebox.v_rotro2qr import v_rotro2qr
from pyvoicebox.v_rotqr2ro import v_rotqr2ro
from pyvoicebox.v_rotqrmean import v_rotqrmean
from pyvoicebox.v_rotro2eu import v_rotro2eu
from pyvoicebox.v_imagehomog import v_imagehomog


def v_rectifyhomog(ims, roc=None, k0=None, mode=''):
    """Apply rectifying homographies to an image set.

    Parameters
    ----------
    ims : list of ndarray
        List of input images.
    roc : ndarray, shape (3, 3) or (3, 3, nc)
        Rotation matrices from world to camera coordinates.
    k0 : float or ndarray
        Camera matrix or focal length.
    mode : str
        Mode string.

    Returns
    -------
    imr : list of ndarray
        Rectified images.
    xa : list of ndarray
        X axis for each image.
    ya : list of ndarray
        Y axis for each image.
    """
    if not isinstance(ims, list):
        ims = [ims]
    nc = len(ims)

    if roc is None:
        roc = np.tile(np.eye(3)[:, :, np.newaxis], (1, 1, nc))
    roc = np.asarray(roc, dtype=float)
    if roc.ndim == 2:
        roc = roc[:, :, np.newaxis]

    if k0 is None:
        k0 = 0.8
    if mode is None:
        mode = ''

    vv = 'v' in mode

    # Determine mean camera orientation
    if 'a' in mode:
        qrc = np.zeros((4, nc))
        for i in range(nc):
            qrc[:, i] = v_rotro2qr(roc[:, :, i])
        rocmean = v_rotqr2ro(v_rotqrmean(qrc)[0])
    else:
        rocmean = np.eye(3)

    modeh = 'kxt' if 'k' in mode else 't'

    imr = []
    xa = []
    ya = []
    for i in range(nc):
        imsz = ims[i].shape
        if np.isscalar(k0) or (isinstance(k0, np.ndarray) and k0.size < 9):
            fe = float(k0) if np.isscalar(k0) else float(k0.ravel()[0])
            if fe < 0.1 * imsz[1]:
                fe = fe * imsz[1]
            xy0 = np.array([(imsz[1] + 1) / 2.0, (imsz[0] + 1) / 2.0])
            k_mat = np.eye(3)
            k_mat[0, 0] = fe
            k_mat[1, 1] = fe
            k_mat[0, 2] = xy0[0]
            k_mat[1, 2] = xy0[1]
        else:
            k_mat = np.asarray(k0, dtype=float).reshape(3, 3)

        rocall = rocmean @ roc[:, :, i].T
        h = k_mat @ rocall @ np.linalg.inv(k_mat)
        img_out, xa_out, ya_out = v_imagehomog(
            np.asarray(ims[i], dtype=np.uint8), h, modeh)
        imr.append(img_out)
        xa.append(xa_out)
        ya.append(ya_out)

    return imr, xa, ya
