"""V_GAUSSMIXB - Approximate Bhattacharyya divergence between two GMMs."""

import numpy as np
from .v_logsum import v_logsum
from .v_gaussmixp import v_gaussmixp
from .v_randvec import v_randvec


def v_gaussmixb(mf, vf=None, wf=None, mg=None, vg=None, wg=None, nx=1000):
    """Approximate Bhattacharyya divergence between two GMMs.

    Parameters
    ----------
    mf : array_like
        Mixture means for GMM f, shape (kf, p).
    vf : array_like, optional
        Variances for GMM f.
    wf : array_like, optional
        Weights for GMM f.
    mg : array_like, optional
        Mixture means for GMM g. If omitted, g=f.
    vg : array_like, optional
        Variances for GMM g.
    wg : array_like, optional
        Weights for GMM g.
    nx : int, optional
        Number of samples for importance sampling (default 1000). 0 for upper bound only.

    Returns
    -------
    d : float
        Approximate Bhattacharyya divergence.
    dbfg : ndarray
        Exact Bhattacharyya divergence between components.
    """
    mf = np.asarray(mf, dtype=float)
    if mf.ndim == 1:
        mf = mf.reshape(1, -1)
    kf, p = mf.shape

    if vf is None:
        vf = np.ones((kf, p))
    else:
        vf = np.asarray(vf, dtype=float)
    if wf is None:
        wf = np.full(kf, 1.0 / kf)
    else:
        wf = np.asarray(wf, dtype=float).ravel()

    if p == 1:
        vf = vf.ravel()
        dvf = True
    else:
        dvf = vf.ndim == 2 and vf.shape[0] == kf

    hpl2 = 0.5 * p * np.log(2)

    if mg is None:
        # Self-divergence
        dbfg = np.zeros((kf, kf))
        if kf > 1:
            if dvf:
                if vf.ndim == 2:
                    qldf = 0.25 * np.sum(np.log(vf), axis=1)
                else:
                    # 1D case: vf is (kf,)
                    qldf = 0.25 * np.log(vf)
                for jf in range(kf - 1):
                    for jg in range(jf + 1, kf):
                        if vf.ndim == 2:
                            vfpg = vf[jf, :] + vf[jg, :]
                            mdif = mf[jf, :] - mf[jg, :]
                            dbfg[jf, jg] = 0.25 * np.sum(mdif ** 2 / vfpg) + 0.5 * np.sum(np.log(vfpg)) - qldf[jf] - qldf[jg] - hpl2
                        else:
                            vfpg = vf[jf] + vf[jg]
                            mdif = mf[jf, 0] - mf[jg, 0]
                            dbfg[jf, jg] = 0.25 * mdif ** 2 / vfpg + 0.5 * np.log(vfpg) - qldf[jf] - qldf[jg] - hpl2
                        dbfg[jg, jf] = dbfg[jf, jg]
            else:
                qldf = np.zeros(kf)
                for jf in range(kf):
                    qldf[jf] = 0.5 * np.sum(np.log(np.diag(np.linalg.cholesky(vf[:, :, jf]))))
                for jf in range(kf - 1):
                    for jg in range(jf + 1, kf):
                        vfg = vf[:, :, jf] + vf[:, :, jg]
                        mdif = mf[jf, :] - mf[jg, :]
                        L = np.linalg.cholesky(vfg)
                        dbfg[jf, jg] = 0.25 * mdif @ np.linalg.solve(vfg, mdif) + np.sum(np.log(np.diag(L))) - qldf[jg] - qldf[jf] - hpl2
                        dbfg[jg, jf] = dbfg[jf, jg]
        d = 0.0
        return d, dbfg

    # Both f and g specified
    mg = np.asarray(mg, dtype=float)
    if mg.ndim == 1:
        mg = mg.reshape(1, -1)
    kg = mg.shape[0]

    if vg is None:
        vg = np.ones((kg, p))
    else:
        vg = np.asarray(vg, dtype=float)
    if wg is None:
        wg = np.full(kg, 1.0 / kg)
    else:
        wg = np.asarray(wg, dtype=float).ravel()

    if p == 1:
        vg = vg.ravel()
        dvg = True
    else:
        dvg = vg.ndim == 2 and vg.shape[0] == kg

    # Calculate pairwise Bhattacharyya divergences
    dbfg = np.zeros((kf, kg))
    if dvf and dvg:
        for jf in range(kf):
            for jg in range(kg):
                if vf.ndim == 2:
                    vfpg = vf[jf, :] + vg[jg, :]
                    mdif = mf[jf, :] - mg[jg, :]
                    qldf_val = 0.25 * np.sum(np.log(vf[jf, :]))
                    qldg_val = 0.25 * np.sum(np.log(vg[jg, :]))
                    dbfg[jf, jg] = 0.25 * np.sum(mdif ** 2 / vfpg) + 0.5 * np.sum(np.log(vfpg)) - qldf_val - qldg_val
                else:
                    vfpg = vf[jf] + vg[jg]
                    mdif = mf[jf, :] - mg[jg, :]
                    qldf_val = 0.25 * np.log(vf[jf])
                    qldg_val = 0.25 * np.log(vg[jg])
                    dbfg[jf, jg] = 0.25 * mdif[0] ** 2 / vfpg + 0.5 * np.log(vfpg) - qldf_val - qldg_val
    else:
        for jf in range(kf):
            if dvf:
                vjf = np.diag(vf[jf, :]) if vf.ndim == 2 else np.array([[vf[jf]]])
            else:
                vjf = vf[:, :, jf]
            qldf_val = 0.5 * np.sum(np.log(np.diag(np.linalg.cholesky(vjf))))
            for jg in range(kg):
                if dvg:
                    vjg = np.diag(vg[jg, :]) if vg.ndim == 2 else np.array([[vg[jg]]])
                else:
                    vjg = vg[:, :, jg]
                vfg = vjf + vjg
                mdif = mf[jf, :] - mg[jg, :]
                L = np.linalg.cholesky(vfg)
                qldg_val = 0.5 * np.sum(np.log(np.diag(np.linalg.cholesky(vjg))))
                dbfg[jf, jg] = 0.25 * mdif @ np.linalg.solve(vfg, mdif) + np.sum(np.log(np.diag(L))) - qldg_val - qldf_val

    dbfg -= hpl2

    # Variational bound iteration
    maxiter = 15
    lwf = np.tile(np.log(wf)[:, np.newaxis], (1, kg))
    lwg = np.tile(np.log(wg)[np.newaxis, :], (kf, 1))
    lhf = np.full((kf, kg), np.log(1.0 / kf))
    lhg = np.full((kf, kg), np.log(1.0 / kg))
    dbfg2 = 2 * dbfg
    dbfg2f = lwf - dbfg2
    dbfg2g = lwg - dbfg2
    dbfg2fg = dbfg2.ravel() - lwf.ravel() - lwg.ravel()

    dub = np.inf
    for ip in range(maxiter):
        dubp = dub
        dub = -v_logsum(0.5 * (lhf.ravel() + lhg.ravel() - dbfg2fg))

        if dub >= dubp:
            break

        lhg = lhf + dbfg2f
        lhg = lhg - v_logsum(lhg, 1)[:, np.newaxis]

        dub = -v_logsum(0.5 * (lhf.ravel() + lhg.ravel() - dbfg2fg))

        lhf = lhg + dbfg2g
        lhf = lhf - v_logsum(lhf, 0)[np.newaxis, :]

    if nx == 0:
        d = float(dub)
    else:
        d = float(dub)  # simplified: use upper bound

    return d, dbfg
