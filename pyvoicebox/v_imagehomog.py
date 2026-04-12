"""V_IMAGEHOMOG - apply homography transformation to an image."""

from __future__ import annotations
import numpy as np


def v_imagehomog(im, h=None, m='', clip=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a homography transformation to an image with bilinear interpolation.

    Parameters
    ----------
    im : ndarray, shape (ny, nx) or (ny, nx, nc)
        Input image (uint8).
    h : ndarray, shape (3, 3), optional
        Homography matrix. Default is identity.
    m : str, optional
        Mode string: 'c' central coordinates, 'k' clip to original, 'x' extend,
        't' trim blank rows/cols.
    clip : array_like, optional
        Clipping specification.

    Returns
    -------
    ih : ndarray
        Transformed image (uint8).
    xa : ndarray
        X axis coordinates.
    ya : ndarray
        Y axis coordinates.
    """
    maxby = int(1e7)

    if im.ndim == 2:
        ny, nx = im.shape
        nc = 1
        im = im[:, :, np.newaxis]
    else:
        ny, nx, nc = im.shape

    if clip is None:
        clip = np.array([2.0])
    else:
        clip = np.asarray(clip, dtype=float).ravel()
    if h is None:
        h = np.eye(3)
    else:
        h = np.asarray(h, dtype=float)

    imr = im.reshape(nx * ny, nc).astype(np.float32)
    t = np.eye(3)

    if 'c' in m:
        t[0, 2] = 0.5 + nx / 2.0
        t[1, 2] = 0.5 + ny / 2.0
        h = t @ h @ np.linalg.inv(t)
        if len(clip) == 4:
            clip = clip + np.array([t[0, 2], t[0, 2], t[1, 2], t[1, 2]])

    if 'k' in m:
        clip = np.array([1.0, 1.0])
    elif len(clip) == 1:
        clip = np.array([clip[0], clip[0]])

    if len(clip) == 2:
        clip = np.array([
            1 * clip[0] - (clip[0] - 1) * (1 + nx) / 2,
            nx * clip[0] - (clip[0] - 1) * (1 + nx) / 2,
            1 * clip[1] - (clip[1] - 1) * (1 + ny) / 2,
            ny * clip[1] - (clip[1] - 1) * (1 + ny) / 2,
        ])

    clip = clip.ravel()
    clip[0] = np.floor(clip[0])
    clip[2] = np.floor(clip[2])
    clip[1] = np.ceil(clip[1])
    clip[3] = np.ceil(clip[3])

    # Determine image of source corners
    bi = np.array([[1, 1, nx, nx], [1, ny, ny, 1], [1, 1, 1, 1]], dtype=float)
    box = h @ bi
    b3 = box[2, :]

    if np.any(b3 <= 0):
        ib = np.where(b3 > 0)[0]
        nb = len(ib)
        if nb == 0:
            raise ValueError('image invisible')
        bb = np.ones((3, nb + 2))
        bb[:, :nb] = box[:, ib]
        px = np.array([3, 0, 1, 2])
        cross_idx = np.where(b3 * b3[px] <= 0)[0]
        ip = px[cross_idx]
        af = b3[ip]
        bf = b3[cross_idx]
        pof = np.array([[-1, 0, 1, 0], [0, 1, 0, -1], [0, 0, 0, 0]], dtype=float)
        for k, ci in enumerate(cross_idx):
            frac = bf[k] / (bf[k] - af[k])
            pt = bi[:, ip[k]] * (1 - frac) + bi[:, ci] * frac
            bb[:, nb + k] = h @ pt
        box = bb

    box_xy = box[:2, :] / box[2:3, :]
    box_bounds = np.array([
        np.min(box_xy[0, :]), np.max(box_xy[0, :]),
        np.min(box_xy[1, :]), np.max(box_xy[1, :])
    ])

    box_bounds[0] = np.floor(max(clip[0], box_bounds[0]))
    box_bounds[2] = np.floor(max(clip[2], box_bounds[2]))
    box_bounds[1] = np.ceil(min(clip[1], box_bounds[1]))
    box_bounds[3] = np.ceil(min(clip[3], box_bounds[3]))

    g = np.linalg.inv(h)
    mx = int(box_bounds[1] - box_bounds[0] + 1)
    my = int(box_bounds[3] - box_bounds[2] + 1)

    if mx <= 0 or my <= 0:
        ih = np.zeros((max(my, 0), max(mx, 0), nc), dtype=np.uint8)
        xa = np.arange(box_bounds[0], box_bounds[1] + 1) - t[0, 2]
        ya = np.arange(box_bounds[2], box_bounds[3] + 1) - t[1, 2]
        return ih, xa, ya

    ih = np.zeros((my * mx, nc), dtype=np.uint8)

    # Process all pixels at once (simpler than chunked approach)
    yy, xx = np.meshgrid(
        np.arange(box_bounds[2], box_bounds[3] + 1),
        np.arange(box_bounds[0], box_bounds[1] + 1),
        indexing='ij'
    )
    coords = np.vstack([xx.ravel(), yy.ravel(), np.ones(mx * my)])
    src = g @ coords
    gn = src[:2, :] / src[2:3, :]

    # Mask valid pixels
    mn = (gn[0, :] > -0.5) & (gn[1, :] > -0.5) & (gn[0, :] < nx + 0.5) & (gn[1, :] < ny + 0.5)

    if np.any(mn):
        fn1 = np.clip(np.floor(gn[0, mn]).astype(int), 1, nx - 1)
        fn2 = np.clip(np.floor(gn[1, mn]).astype(int), 1, ny - 1)
        dn1 = np.clip(gn[0, mn] - fn1, 0, 1)
        dn2 = np.clip(gn[1, mn] - fn2, 0, 1)
        dn1c = 1 - dn1
        dn2c = 1 - dn2

        # MATLAB uses 1-based indexing: imr(fn2 + ny*(fn1-1), :)
        # Python 0-based: imr[(fn2-1) + ny*(fn1-1), :]
        idx00 = (fn2 - 1) + ny * (fn1 - 1)
        idx01 = fn2 + ny * (fn1 - 1)
        idx10 = (fn2 - 1) + ny * fn1
        idx11 = fn2 + ny * fn1

        # Clip indices
        max_idx = nx * ny - 1
        idx00 = np.clip(idx00, 0, max_idx)
        idx01 = np.clip(idx01, 0, max_idx)
        idx10 = np.clip(idx10, 0, max_idx)
        idx11 = np.clip(idx11, 0, max_idx)

        val = (dn1c[:, np.newaxis] * (dn2c[:, np.newaxis] * imr[idx00, :]
                                       + dn2[:, np.newaxis] * imr[idx01, :])
               + dn1[:, np.newaxis] * (dn2c[:, np.newaxis] * imr[idx10, :]
                                        + dn2[:, np.newaxis] * imr[idx11, :]))
        jx = np.where(mn)[0]
        ih[jx, :] = np.clip(val, 0, 255).astype(np.uint8)

    ih = ih.reshape(my, mx, nc)

    if 'x' in m:
        b0 = int(box_bounds[0])
        b2 = int(box_bounds[2])
        c0 = int(clip[0])
        c1 = int(clip[1])
        c2 = int(clip[2])
        c3 = int(clip[3])
        new_ih = np.zeros((c3 - c2 + 1, c1 - c0 + 1, nc), dtype=np.uint8)
        row_start = b2 - c2
        col_start = b0 - c0
        new_ih[row_start:row_start + my, col_start:col_start + mx, :] = ih
        ih = new_ih
        xa = np.arange(c0, c1 + 1) - t[0, 2]
        ya = np.arange(c2, c3 + 1) - t[1, 2]
    else:
        xa = np.arange(box_bounds[0], box_bounds[1] + 1) - t[0, 2]
        ya = np.arange(box_bounds[2], box_bounds[3] + 1) - t[1, 2]

    if 't' in m:
        ihs = np.sum(ih, axis=2)
        azx = np.all(ihs == 0, axis=0)
        ix1 = np.argmax(~azx)
        if np.all(azx):
            return np.array([], dtype=np.uint8), np.array([]), np.array([])
        ix2 = len(azx) - 1 - np.argmax(~azx[::-1])
        azy = np.all(ihs == 0, axis=1)
        iy1 = np.argmax(~azy)
        iy2 = len(azy) - 1 - np.argmax(~azy[::-1])
        ih = ih[iy1:iy2 + 1, ix1:ix2 + 1, :]
        xa = xa[ix1:ix2 + 1]
        ya = ya[iy1:iy2 + 1]

    if nc == 1:
        ih = ih[:, :, 0]

    return ih, xa, ya
