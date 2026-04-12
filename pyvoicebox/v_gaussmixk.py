"""V_GAUSSMIXK - Approximate KL divergence between two GMMs."""

from __future__ import annotations
import numpy as np
from .v_logsum import v_logsum


def v_gaussmixk(mf, vf=None, wf=None, mg=None, vg=None, wg=None) -> tuple[float, np.ndarray]:
    """Approximate Kullback-Leibler divergence between two GMMs.

    Parameters
    ----------
    mf : array_like
        Mixture means for GMM f, shape (kf, p).
    vf : array_like, optional
        Variances for GMM f. Diagonal (kf, p) or full (p, p, kf).
    wf : array_like, optional
        Weights for GMM f.
    mg : array_like, optional
        Mixture means for GMM g.
    vg : array_like, optional
        Variances for GMM g.
    wg : array_like, optional
        Weights for GMM g.

    Returns
    -------
    d : float
        Approximate KL divergence D(f||g).
    klfg : ndarray
        Exact KL divergence between components.
    """
    mf = np.asarray(mf, dtype=float)
    if mf.ndim == 1:
        mf = mf.reshape(1, -1)
    kf, p = mf.shape

    if wf is None or len(np.asarray(wf).ravel()) == 0:
        wf = np.full(kf, 1.0 / kf)
    else:
        wf = np.asarray(wf, dtype=float).ravel()

    if vf is None or len(np.asarray(vf).ravel()) == 0:
        vf = np.ones((kf, p))
    else:
        vf = np.asarray(vf, dtype=float)

    fvf = vf.ndim > 2 or (vf.ndim == 2 and vf.shape[0] > kf)

    # Calculate intra-f KL divergences
    klff = np.zeros((kf, kf))
    ixdp = np.arange(0, p * p, p + 1)  # diagonal indices

    if fvf:
        dvf = np.zeros(kf)
        for i in range(kf):
            dvf[i] = np.log(np.linalg.det(vf[:, :, i]))
        for j in range(kf):
            pfj = np.linalg.inv(vf[:, :, j])
            for i in range(kf):
                if i == j:
                    continue
                mffj = mf[i, :] - mf[j, :]
                pfjvf = pfj @ vf[:, :, i]
                klff[i, j] = 0.5 * (dvf[j] - p - dvf[i] + np.trace(pfjvf) + mffj @ pfj @ mffj)
    else:
        dvf = np.log(np.prod(vf, axis=1))
        pf = 1.0 / vf
        for j in range(kf):
            for i in range(kf):
                if i == j:
                    continue
                mffj = mf[i, :] - mf[j, :]
                klff[i, j] = 0.5 * (dvf[j] - dvf[i] + np.sum(vf[i, :] * pf[j, :]) - p + np.sum(mffj ** 2 * pf[j, :]))

    if mg is None:
        d = 0.0
        klfg = klff
    else:
        mg = np.asarray(mg, dtype=float)
        if mg.ndim == 1:
            mg = mg.reshape(1, -1)
        kg = mg.shape[0]

        if vg is None or len(np.asarray(vg).ravel()) == 0:
            vg = np.ones((kg, p))
        else:
            vg = np.asarray(vg, dtype=float)

        if wg is None or len(np.asarray(wg).ravel()) == 0:
            wg = np.full(kg, 1.0 / kg)
        else:
            wg = np.asarray(wg, dtype=float).ravel()

        fvg = vg.ndim > 2 or (vg.ndim == 2 and vg.shape[0] > kg)

        klfg = np.zeros((kf, kg))

        if fvg:
            dvg = np.zeros(kg)
            for j in range(kg):
                dvg[j] = np.log(np.linalg.det(vg[:, :, j]))
            if fvf:
                for j in range(kg):
                    pgj = np.linalg.inv(vg[:, :, j])
                    for i in range(kf):
                        mfgj = mf[i, :] - mg[j, :]
                        pgjvf = pgj @ vf[:, :, i]
                        klfg[i, j] = 0.5 * (dvg[j] - p - dvf[i] + np.trace(pgjvf) + mfgj @ pgj @ mfgj)
            else:
                for j in range(kg):
                    pgj = np.linalg.inv(vg[:, :, j])
                    for i in range(kf):
                        mfgj = mf[i, :] - mg[j, :]
                        klfg[i, j] = 0.5 * (dvg[j] - p - dvf[i] + np.sum(vf[i, :] * np.diag(pgj)) + mfgj @ pgj @ mfgj)
        else:
            dvg = np.log(np.prod(vg, axis=1))
            pg = 1.0 / vg
            if fvf:
                for j in range(kg):
                    for i in range(kf):
                        mfgj = mf[i, :] - mg[j, :]
                        klfg[i, j] = 0.5 * (dvg[j] - dvf[i] + np.sum(np.diag(vf[:, :, i]) * pg[j, :]) - p + np.sum(mfgj ** 2 * pg[j, :]))
            else:
                for j in range(kg):
                    for i in range(kf):
                        mfgj = mf[i, :] - mg[j, :]
                        klfg[i, j] = 0.5 * (dvg[j] - dvf[i] + np.sum(vf[i, :] * pg[j, :]) - p + np.sum(mfgj ** 2 * pg[j, :]))

        # Calculate variational approximation
        d = wf @ (v_logsum(-klff, 1, wf) - v_logsum(-klfg, 1, wg))

    return d, klfg
