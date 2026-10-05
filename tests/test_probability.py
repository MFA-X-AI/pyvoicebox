"""Tests for Gaussian mixture and probability functions."""

import os
import numpy as np
import scipy.io as sio
import pytest

REF_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pyvoicebox-test', 'ref_data')


def load_ref(name):
    return sio.loadmat(os.path.join(REF_DIR, name), squeeze_me=True)


# ============================================================
# v_berk2prob / v_prob2berk
# ============================================================
class TestBerk:
    def test_berk2prob(self):
        from pyvoicebox.v_berk2prob import v_berk2prob
        ref = load_ref('ref_berk.mat')
        p, d = v_berk2prob(ref['b_test'])
        np.testing.assert_allclose(p, ref['p_berk'], rtol=1e-12)
        np.testing.assert_allclose(d, ref['d_berk'], rtol=1e-12)

    def test_prob2berk(self):
        from pyvoicebox.v_prob2berk import v_prob2berk
        ref = load_ref('ref_berk.mat')
        b, d = v_prob2berk(ref['p_berk'])
        np.testing.assert_allclose(b, ref['b_back'], rtol=1e-12)
        np.testing.assert_allclose(d, ref['d_back'], rtol=1e-12)

    def test_roundtrip(self):
        from pyvoicebox.v_berk2prob import v_berk2prob
        from pyvoicebox.v_prob2berk import v_prob2berk
        b_orig = np.array([-5, -2, 0, 2, 5])
        p, _ = v_berk2prob(b_orig)
        b_back, _ = v_prob2berk(p)
        np.testing.assert_allclose(b_back, b_orig, rtol=1e-12)


# ============================================================
# v_normcdflog
# ============================================================
class TestNormcdflog:
    def test_basic(self):
        from pyvoicebox.v_normcdflog import v_normcdflog
        ref = load_ref('ref_normcdflog.mat')
        p = v_normcdflog(ref['x_ncl'])
        np.testing.assert_allclose(p, ref['p_ncl'], rtol=1e-10)

    def test_with_mean_std(self):
        from pyvoicebox.v_normcdflog import v_normcdflog
        ref = load_ref('ref_normcdflog.mat')
        p = v_normcdflog(ref['x_ncl'], 2, 3)
        np.testing.assert_allclose(p, ref['p_ncl2'], rtol=1e-10)


# ============================================================
# v_vonmisespdf
# ============================================================
class TestVonMises:
    def test_basic(self):
        from pyvoicebox.v_vonmisespdf import v_vonmisespdf
        ref = load_ref('ref_vonmises.mat')
        p = v_vonmisespdf(ref['x_vm'], 0, 2)
        np.testing.assert_allclose(p, ref['p_vm'], rtol=1e-12)

    def test_shifted(self):
        from pyvoicebox.v_vonmisespdf import v_vonmisespdf
        ref = load_ref('ref_vonmises.mat')
        p = v_vonmisespdf(ref['x_vm'], 1, 5)
        np.testing.assert_allclose(p, ref['p_vm2'], rtol=1e-12)


# ============================================================
# v_besselratio
# ============================================================
class TestBesselratio:
    def test_order0(self):
        from pyvoicebox.v_besselratio import v_besselratio
        ref = load_ref('ref_besselratio.mat')
        y = v_besselratio(ref['x_br'], 0, 10)
        np.testing.assert_allclose(y, ref['y_br'], rtol=1e-10)

    def test_order1(self):
        from pyvoicebox.v_besselratio import v_besselratio
        ref = load_ref('ref_besselratio.mat')
        y = v_besselratio(ref['x_br'], 1, 10)
        np.testing.assert_allclose(y, ref['y_br1'], rtol=1e-10)

    def test_edge_zero(self):
        from pyvoicebox.v_besselratio import v_besselratio
        y = v_besselratio(np.array([0.0]), 0, 5)
        np.testing.assert_allclose(y, [0.0], atol=1e-15)

    def test_edge_inf(self):
        from pyvoicebox.v_besselratio import v_besselratio
        y = v_besselratio(np.array([np.inf]), 0, 5)
        np.testing.assert_allclose(y, [1.0], atol=1e-15)

    def test_monotonic(self):
        from pyvoicebox.v_besselratio import v_besselratio
        x = np.array([0, 0.1, 1, 10, 100, np.inf])
        y = v_besselratio(x, 0, 10)
        assert np.all(np.diff(y) >= 0)


# ============================================================
# v_besselratioi
# ============================================================
class TestBesselratioi:
    def test_basic(self):
        from pyvoicebox.v_besselratioi import v_besselratioi
        ref = load_ref('ref_besselratioi.mat')
        s = v_besselratioi(ref['r_bri'], 0, 10)
        np.testing.assert_allclose(s, ref['s_bri'], rtol=1e-6)


# ============================================================
# v_besratinv0
# ============================================================
class TestBesratinv0:
    def test_basic(self):
        from pyvoicebox.v_besratinv0 import v_besratinv0
        ref = load_ref('ref_besratinv0.mat')
        k = v_besratinv0(ref['r_bi0'])
        np.testing.assert_allclose(k, ref['k_bi0'], rtol=1e-6)

    def test_roundtrip(self):
        from pyvoicebox.v_besselratio import v_besselratio
        from pyvoicebox.v_besratinv0 import v_besratinv0
        r_orig = np.array([0.2, 0.5, 0.8])
        k = v_besratinv0(r_orig)
        r_back = v_besselratio(k, 0, 10)
        np.testing.assert_allclose(r_back, r_orig, rtol=1e-5)


# ============================================================
# v_chimv
# ============================================================
class TestChimv:
    def test_n1(self):
        from pyvoicebox.v_chimv import v_chimv
        ref = load_ref('ref_chimv.mat')
        m, v = v_chimv(1, 0, 1)
        np.testing.assert_allclose(m, ref['m_chi1'], rtol=1e-10)
        np.testing.assert_allclose(v, ref['v_chi1'], rtol=1e-10)

    def test_n3(self):
        from pyvoicebox.v_chimv import v_chimv
        ref = load_ref('ref_chimv.mat')
        m, v = v_chimv(3, 2, 1)
        np.testing.assert_allclose(m, ref['m_chi2'], rtol=1e-3)
        np.testing.assert_allclose(v, ref['v_chi2'], rtol=1e-2)

    def test_n5_vector(self):
        from pyvoicebox.v_chimv import v_chimv
        ref = load_ref('ref_chimv.mat')
        m, v = v_chimv(5, np.array([0, 1, 3]), 2)
        np.testing.assert_allclose(m, ref['m_chi3'], rtol=1e-3)
        np.testing.assert_allclose(v, ref['v_chi3'], rtol=1e-2)


# ============================================================
# v_maxgauss
# ============================================================
class TestMaxgauss:
    def test_basic(self):
        from pyvoicebox.v_maxgauss import v_maxgauss
        ref = load_ref('ref_maxgauss.mat')
        u, v, p, _ = v_maxgauss(ref['m_mg'], ref['c_mg'])
        np.testing.assert_allclose(u, ref['u_mg'], rtol=1e-10)
        np.testing.assert_allclose(v, ref['v_mg'], rtol=1e-8)
        np.testing.assert_allclose(p, ref['p_mg'].ravel(), rtol=1e-10)


# ============================================================
# v_gausprod
# ============================================================
class TestGausprod:
    def test_scalar_cov(self):
        from pyvoicebox.v_gausprod import v_gausprod
        ref = load_ref('ref_gausprod.mat')
        g, u, k = v_gausprod(ref['m_gp1'], ref['c_gp1'])
        np.testing.assert_allclose(g, ref['g_gp1'], rtol=1e-10)
        np.testing.assert_allclose(u, ref['u_gp1'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(k, float(ref['k_gp1']), rtol=1e-10)


# ============================================================
# v_gaussmixp (CRITICAL - 22 callers)
# ============================================================
class TestGaussmixp:
    def test_diagonal(self):
        from pyvoicebox.v_gaussmixp import v_gaussmixp
        ref = load_ref('ref_gaussmixp.mat')
        lp, rp, kh, kp = v_gaussmixp(
            ref['y_gmp'], ref['m_gmp'], ref['v_gmp'], ref['w_gmp']
        )
        np.testing.assert_allclose(lp, ref['lp_gmp'], rtol=1e-10)
        np.testing.assert_allclose(rp, ref['rp_gmp'], rtol=1e-10)
        # MATLAB kh is 1-based, Python is 0-based
        np.testing.assert_array_equal(kh, ref['kh_gmp'].ravel() - 1)
        np.testing.assert_allclose(kp, ref['kp_gmp'].ravel(), rtol=1e-10)

    def test_full_covariance(self):
        from pyvoicebox.v_gaussmixp import v_gaussmixp
        ref = load_ref('ref_gaussmixp_full.mat')
        lp, rp, kh, kp = v_gaussmixp(
            ref['y_gmpf'], ref['m_gmpf'], ref['v_gmpf'], ref['w_gmpf']
        )
        np.testing.assert_allclose(lp, ref['lp_gmpf'], rtol=1e-10)
        np.testing.assert_allclose(rp, ref['rp_gmpf'], rtol=1e-10)
        np.testing.assert_array_equal(kh, ref['kh_gmpf'].ravel() - 1)
        np.testing.assert_allclose(kp, ref['kp_gmpf'].ravel(), rtol=1e-10)

    def test_single_gaussian(self):
        """Test with a single Gaussian (k=1)."""
        from pyvoicebox.v_gaussmixp import v_gaussmixp
        from scipy.stats import multivariate_normal
        m = np.array([[1.0, 2.0]])
        v = np.array([[0.5, 1.0]])
        w = np.array([1.0])
        y = np.array([[1.0, 2.0], [0.0, 0.0], [2.0, 3.0]])
        lp, rp, kh, kp = v_gaussmixp(y, m, v, w)
        # Compare with scipy
        expected = multivariate_normal.logpdf(y, mean=m[0], cov=np.diag(v[0]))
        np.testing.assert_allclose(lp, expected, rtol=1e-10)


# ============================================================
# v_gaussmixk
# ============================================================
class TestGaussmixk:
    def test_two_gmms(self):
        from pyvoicebox.v_gaussmixk import v_gaussmixk
        ref = load_ref('ref_gaussmixk.mat')
        d, klfg = v_gaussmixk(
            ref['m_kl1'], ref['v_kl1'], ref['w_kl1'],
            ref['m_kl2'], ref['v_kl2'], ref['w_kl2']
        )
        np.testing.assert_allclose(d, ref['d_kl'], rtol=1e-10)
        np.testing.assert_allclose(klfg, ref['klfg_kl'], rtol=1e-10)

    def test_self(self):
        from pyvoicebox.v_gaussmixk import v_gaussmixk
        ref = load_ref('ref_gaussmixk.mat')
        d, klfg = v_gaussmixk(ref['m_kl1'], ref['v_kl1'], ref['w_kl1'])
        np.testing.assert_allclose(d, ref['d_kl_self'], rtol=1e-10)
        np.testing.assert_allclose(klfg, ref['klfg_self'], rtol=1e-10)


# ============================================================
# v_gaussmixg
# ============================================================
class TestGaussmixg:
    def test_mean_var(self):
        from pyvoicebox.v_gaussmixg import v_gaussmixg
        ref = load_ref('ref_gaussmixg.mat')
        mg, vg, pg, pv = v_gaussmixg(ref['m_gg'], ref['v_gg'], ref['w_gg'], 2)
        np.testing.assert_allclose(mg, ref['mg_gg'], rtol=1e-10)
        np.testing.assert_allclose(vg, ref['vg_gg'], rtol=1e-10)

    def test_modes(self):
        from pyvoicebox.v_gaussmixg import v_gaussmixg
        ref = load_ref('ref_gaussmixg.mat')
        mg, vg, pg, pv = v_gaussmixg(ref['m_gg'], ref['v_gg'], ref['w_gg'], 2)
        # Modes should be near the mixture means for well-separated Gaussians
        ref_pg = ref['pg_gg']
        if ref_pg.ndim == 1:
            ref_pg = ref_pg.reshape(1, -1)
        # Sort both by first coordinate to compare
        idx_py = np.argsort(pg[:, 0])
        idx_ref = np.argsort(ref_pg[:, 0])
        np.testing.assert_allclose(pg[idx_py], ref_pg[idx_ref], rtol=1e-3, atol=0.1)


# ============================================================
# v_gaussmixt
# ============================================================
class TestGaussmixt:
    def test_1d(self):
        from pyvoicebox.v_gaussmixt import v_gaussmixt
        ref = load_ref('ref_gaussmixt.mat')
        m, v, w = v_gaussmixt(
            ref['m_t1'].reshape(-1, 1), ref['v_t1'].reshape(-1, 1),
            ref['w_t1'], ref['m_t2'].reshape(-1, 1),
            ref['v_t2'].reshape(-1, 1), ref['w_t2']
        )
        np.testing.assert_allclose(m.ravel(), ref['m_tp'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(v.ravel(), ref['v_tp'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(w.ravel(), ref['w_tp'].ravel(), rtol=1e-10)


# ============================================================
# v_gaussmixm
# ============================================================
class TestGaussmixm:
    def test_2d(self):
        from pyvoicebox.v_gaussmixm import v_gaussmixm
        ref = load_ref('ref_gaussmixm.mat')
        mm, mc = v_gaussmixm(ref['m_gm'], ref['v_gm'], ref['w_gm'], ref['z_gm'].reshape(1, -1))
        # Loose tolerances: v_gaussmixm approximates each mixture's magnitude
        # as a Nakagami-m distribution via moment matching, which is inexact
        # for P>1 with nonzero means (see MATLAB source comment).
        np.testing.assert_allclose(mm, ref['mm_gm'], rtol=1e-3)
        np.testing.assert_allclose(mc, ref['mc_gm'], rtol=1e-2)

    @pytest.mark.parametrize('origins', [[0.0], [-1.0, 2.0]])
    def test_1d_mixture_magnitude_moments(self, origins):
        from scipy.integrate import quad
        from scipy.stats import norm
        from pyvoicebox.v_gaussmixm import v_gaussmixm

        means = np.array([0.0, 1.5])
        variances = np.array([1.0, 2.0])
        weights = np.array([0.3, 0.7])
        origins = np.asarray(origins)
        mm, mc = v_gaussmixm(means[:, None], variances[:, None],
                             weights, origins[:, None])

        def pdf(x):
            return weights @ norm.pdf(x, means, np.sqrt(variances))

        edges = np.r_[-np.inf, origins, np.inf]

        def expectation(fn):
            return sum(quad(lambda x: fn(x) * pdf(x), left, right)[0]
                       for left, right in zip(edges[:-1], edges[1:]))

        expected_mean = np.array([
            expectation(lambda x: abs(x - z)) for z in origins
        ])
        expected_second = np.array([
            [expectation(lambda x: abs(x - zi) * abs(x - zj))
             for zj in origins] for zi in origins
        ])
        np.testing.assert_allclose(mm, expected_mean, rtol=1e-10)
        np.testing.assert_allclose(mc, expected_second - np.outer(expected_mean, expected_mean),
                                   rtol=1e-10, atol=1e-10)


# ============================================================
# v_gaussmixb
# ============================================================
class TestGaussmixb:
    def test_self(self):
        from pyvoicebox.v_gaussmixb import v_gaussmixb
        ref = load_ref('ref_gaussmixb.mat')
        d, dbfg = v_gaussmixb(
            ref['m_b1'].reshape(-1, 1), ref['v_b1'].reshape(-1, 1),
            ref['w_b1']
        )
        np.testing.assert_allclose(d, ref['d_bself'], rtol=1e-10)
        np.testing.assert_allclose(dbfg, ref['dbfg_self'], rtol=1e-10)

    def test_two_gmms_upper_bound(self):
        from pyvoicebox.v_gaussmixb import v_gaussmixb
        ref = load_ref('ref_gaussmixb.mat')
        d, _ = v_gaussmixb(
            ref['m_b1'].reshape(-1, 1), ref['v_b1'].reshape(-1, 1), ref['w_b1'],
            ref['m_b2'].reshape(-1, 1), ref['v_b2'].reshape(-1, 1), ref['w_b2'], 0
        )
        np.testing.assert_allclose(d, ref['d_b'], rtol=1e-6)


# ============================================================
# v_histndim
# ============================================================
class TestHistndim:
    def test_2d(self):
        from pyvoicebox.v_histndim import v_histndim
        ref = load_ref('ref_histndim.mat')
        x_hist = ref['x_hist']
        b_hist = ref['b_hist'].reshape(-1, 1)
        v_out, t_out = v_histndim(x_hist, b_hist)
        # Compare shapes and total count
        ref_v = ref['v_hist']
        assert v_out.shape == ref_v.shape, f"Shape mismatch: {v_out.shape} vs {ref_v.shape}"
        np.testing.assert_allclose(np.sum(v_out), np.sum(ref_v), rtol=1e-10)


# ============================================================
# v_kmeans (test with known data, deterministic)
# ============================================================
class TestKmeans:
    def test_simple_clustering(self):
        from pyvoicebox.v_kmeans import v_kmeans
        np.random.seed(42)
        # Create well-separated clusters
        c1 = np.random.randn(50, 2) + np.array([0, 0])
        c2 = np.random.randn(50, 2) + np.array([10, 10])
        data = np.vstack([c1, c2])
        np.random.seed(42)
        x, g, j, gg = v_kmeans(data, 2)
        # All points in c1 should be in one cluster, c2 in another
        assert len(np.unique(j[:50])) == 1
        assert len(np.unique(j[50:])) == 1
        assert j[0] != j[50]


# ============================================================
# v_kmeanlbg
# ============================================================
class TestKmeanlbg:
    def test_simple(self):
        from pyvoicebox.v_kmeanlbg import v_kmeanlbg
        np.random.seed(42)
        c1 = np.random.randn(50, 2) + np.array([0, 0])
        c2 = np.random.randn(50, 2) + np.array([10, 10])
        data = np.vstack([c1, c2])
        np.random.seed(42)
        x, esq, j = v_kmeanlbg(data, 2)
        assert x.shape == (2, 2)
        assert len(np.unique(j[:50])) == 1


# ============================================================
# v_kmeanhar
# ============================================================
class TestKmeanhar:
    def test_simple(self):
        from pyvoicebox.v_kmeanhar import v_kmeanhar
        np.random.seed(42)
        c1 = np.random.randn(50, 2) + np.array([0, 0])
        c2 = np.random.randn(50, 2) + np.array([10, 10])
        data = np.vstack([c1, c2])
        # Provide initial centres in different clusters for reliable convergence
        x0 = np.array([[0.0, 0.0], [10.0, 10.0]])
        x, g, xn, gg = v_kmeanhar(data, 2, x0=x0)
        assert x.shape == (2, 2)
        # Centres should be near (0,0) and (10,10)
        centres = np.sort(x[:, 0])
        assert centres[0] < 2.0
        assert centres[1] > 8.0


# ============================================================
# v_randiscr
# ============================================================
class TestRandiscr:
    def test_uniform(self):
        from pyvoicebox.v_randiscr import v_randiscr
        np.random.seed(42)
        x = v_randiscr(np.array([1, 1, 1, 1]), 1000)
        # Should have roughly equal counts
        for i in range(4):
            count = np.sum(x == i)
            assert 150 < count < 350, f"Count for {i}: {count}"

    def test_weighted(self):
        from pyvoicebox.v_randiscr import v_randiscr
        np.random.seed(42)
        x = v_randiscr(np.array([9, 1]), 1000)
        count_0 = np.sum(x == 0)
        assert count_0 > 800  # Should be about 900


# ============================================================
# v_randvec
# ============================================================
class TestRandvec:
    def test_single_gaussian(self):
        from pyvoicebox.v_randvec import v_randvec
        np.random.seed(42)
        m = np.array([[5.0, 10.0]])
        c = np.array([[1.0, 4.0]])
        x, kx = v_randvec(10000, m, c)
        np.testing.assert_allclose(np.mean(x, axis=0), m[0], atol=0.1)
        np.testing.assert_allclose(np.var(x, axis=0), c[0], atol=0.2)


# ============================================================
# v_gmmlpdf (wrapper)
# ============================================================
class TestGmmlpdf:
    def test_wrapper(self):
        from pyvoicebox.v_gmmlpdf import v_gmmlpdf
        from pyvoicebox.v_gaussmixp import v_gaussmixp
        m = np.array([[0.0, 0.0]])
        v = np.array([[1.0, 1.0]])
        w = np.array([1.0])
        y = np.array([[0.0, 0.0], [1.0, 1.0]])
        lp1 = v_gmmlpdf(y, m, v, w)[0]
        lp2 = v_gaussmixp(y, m, v, w)[0]
        np.testing.assert_array_equal(lp1, lp2)


# ============================================================
# v_disteusq
# ============================================================
class TestDisteusq:
    def test_full_matrix(self):
        from pyvoicebox.v_disteusq import v_disteusq
        x = np.array([[0, 0], [1, 0], [0, 1]])
        y = np.array([[1, 1], [0, 0]])
        d = v_disteusq(x, y, 'x')
        expected = np.array([[2, 0], [1, 1], [1, 1]])
        np.testing.assert_allclose(d, expected, rtol=1e-12)

    def test_pairwise(self):
        from pyvoicebox.v_disteusq import v_disteusq
        x = np.array([[0, 0], [3, 4]])
        y = np.array([[1, 0], [0, 0]])
        d = v_disteusq(x, y, 'd')
        np.testing.assert_allclose(d, [1, 25], rtol=1e-12)


# ============================================================
# v_gaussmix (EM fitting)
# ============================================================
class TestGaussmix:
    def test_basic_em(self):
        from pyvoicebox.v_gaussmix import v_gaussmix
        np.random.seed(42)
        n = 200
        c1 = np.random.randn(n, 2) * 0.5 + np.array([0, 0])
        c2 = np.random.randn(n, 2) * 0.5 + np.array([5, 5])
        x = np.vstack([c1, c2])
        np.random.seed(42)
        m, v, w, g, f, pp, gg = v_gaussmix(x, None, None, 2, 'f')
        # Check that we found 2 distinct centres near (0,0) and (5,5)
        centres = np.sort(m[:, 0])
        assert centres[0] < 2.0, f"First centre too high: {centres[0]}"
        assert centres[1] > 3.0, f"Second centre too low: {centres[1]}"
        # Weights should be roughly equal
        assert np.all(w > 0.3)
        assert np.all(w < 0.7)

    @pytest.mark.parametrize('initialization', ['hv', 'explicit', 'diagonal_full'])
    def test_full_covariance_single_component(self, initialization):
        from pyvoicebox.v_gaussmix import v_gaussmix
        from pyvoicebox.v_gaussmixp import v_gaussmixp

        x = np.array([[1., 1.], [2., 2.], [3., 4.], [4., 3.]])
        if initialization == 'hv':
            args = dict(m0=1, v0='hv')
        else:
            covariance = (np.array([[1., .5], [.5, 1.]])
                          if initialization == 'explicit' else np.eye(2))
            args = dict(m0=[[0., 0.]], v0=covariance[:, :, None], w0=[1.])
        m, v, w, g, f, pp, gg = v_gaussmix(x, l=10, **args)
        np.testing.assert_allclose(m, [[2.5, 2.5]], atol=1e-12)
        np.testing.assert_allclose(v[:, :, 0], [[1.25, 1.], [1., 1.25]], atol=1e-12)
        np.testing.assert_allclose(w, [1.])
        np.testing.assert_allclose(pp, v_gaussmixp(x, m, v, w)[0], atol=1e-12)
        np.testing.assert_allclose(g, pp.mean(), atol=1e-12)
        np.testing.assert_allclose(gg[-1], g, atol=1e-12)
        assert f == 0

    def test_weighted_full_covariance(self):
        from pyvoicebox.v_gaussmix import v_gaussmix
        from pyvoicebox.v_gaussmixp import v_gaussmixp

        x = np.array([[10., 1.], [20., 2.], [30., 4.], [40., 3.]])
        weights = np.array([1., 2., 3., 4.])
        normalized = weights / weights.sum()
        expected_mean = normalized @ x
        centered = x - expected_mean
        expected_covariance = centered.T @ (centered * normalized[:, None])
        m, v, w, g, _, pp, gg = v_gaussmix(
            x, l=10, m0=[[0., 0.]], v0=np.array([[20., 1.], [1., 2.]]),
            w0=[1.], wx=weights,
        )
        np.testing.assert_allclose(m[0], expected_mean, atol=1e-12)
        np.testing.assert_allclose(v[:, :, 0], expected_covariance, atol=1e-12)
        np.testing.assert_allclose(pp, v_gaussmixp(x, m, v, w)[0], atol=1e-12)
        np.testing.assert_allclose(g, normalized @ pp, atol=1e-12)
        np.testing.assert_allclose(gg[-1], g, atol=1e-12)

    @pytest.mark.parametrize('initialization', ['explicit', 'kmv'])
    def test_correlated_two_component_fit(self, initialization):
        from pyvoicebox.v_gaussmix import v_gaussmix
        from pyvoicebox.v_gaussmixp import v_gaussmixp

        rng = np.random.RandomState(17)
        c1 = rng.multivariate_normal([-3., 5.], [[1.5, .9], [.9, 1.]], size=160)
        c2 = rng.multivariate_normal([5., -2.], [[.7, -.5], [-.5, 1.2]], size=240)
        x = np.vstack([c1, c2])
        means = np.array([[-2., 4.], [4., -1.]])
        if initialization == 'explicit':
            covariance = np.repeat(np.array([[3., .2], [.2, 3.]])[:, :, None], 2, axis=2)
            args = dict(m0=means, v0=covariance, w0=[.5, .5])
        else:
            args = dict(m0=means, v0=initialization)
        m, v, w, g, f, pp, gg = v_gaussmix(x, l=30, **args)
        order = np.argsort(m[:, 0])
        np.testing.assert_allclose(m[order], [c1.mean(axis=0), c2.mean(axis=0)], atol=1e-4)
        np.testing.assert_allclose(w[order], [.4, .6], atol=1e-4)
        for idx, cloud in zip(order, [c1, c2]):
            np.testing.assert_allclose(v[:, :, idx], np.cov(cloud, rowvar=False, bias=True), atol=1e-4)
            assert np.linalg.eigvalsh(v[:, :, idx]).min() > 0
        np.testing.assert_allclose(pp, v_gaussmixp(x, m, v, w)[0], atol=1e-11)
        np.testing.assert_allclose(g, pp.mean(), atol=1e-12)
        np.testing.assert_allclose(gg[-1], g, atol=1e-12)
        assert np.all(np.diff(gg) >= -1e-10)
        assert f > 0

    def test_full_covariance_eigenvalue_floor(self):
        from pyvoicebox.v_gaussmix import v_gaussmix
        from pyvoicebox.v_gaussmixp import v_gaussmixp

        x = np.array([[1., 2.], [2., 4.], [3., 6.], [4., 8.]])
        m, v, w, _, _, pp, _ = v_gaussmix(
            x, c=.05, l=10, m0=[[0., 0.]],
            v0=np.array([[1., 1.], [1., 1.]])[:, :, None], w0=[1.],
        )
        scaled_covariance = v[:, :, 0] / np.outer(x.std(axis=0), x.std(axis=0))
        np.testing.assert_allclose(np.linalg.eigvalsh(scaled_covariance), [.05, 2.], atol=1e-12)
        np.testing.assert_allclose(pp, v_gaussmixp(x, m, v, w)[0], atol=1e-12)

    def test_explicit_diagonal_variance_normalization(self):
        from pyvoicebox.v_gaussmix import v_gaussmix
        from pyvoicebox.v_gaussmixp import v_gaussmixp

        x = np.array([[10., 1.], [20., 2.], [30., 4.], [40., 3.]])
        mean, variance = np.array([[5., 1.]]), np.array([[20., 2.]])
        m, v, w, _, _, pp, _ = v_gaussmix(x, l=0, m0=mean, v0=variance, w0=[1.])
        np.testing.assert_allclose(m, mean, atol=1e-12)
        np.testing.assert_allclose(v, variance, atol=1e-12)
        np.testing.assert_allclose(pp, v_gaussmixp(x, mean, variance, [1.])[0], atol=1e-12)


class TestProbabilityRegressions:
    @pytest.mark.parametrize('mode, moments, expected', [
        ('mR', [2., 1., 0., 3.], [2., 5., 14., 43.]),
        ('rM', [2., 5., 14., 43.], [2., 1., 0., 3.]),
    ])
    def test_translated_moments(self, mode, moments, expected):
        from pyvoicebox.v_pdfmoments import v_pdfmoments

        converted, _, _ = v_pdfmoments(mode, moments)
        np.testing.assert_allclose(converted, expected, atol=1e-12)

    def test_affine_moment_transform(self):
        from pyvoicebox.v_pdfmoments import v_pdfmoments

        raw, _, _ = v_pdfmoments('mR', [2., 1., 0., 3.], b=3., a=2.)
        np.testing.assert_allclose(raw, [7., 53., 427., 3625.], atol=1e-12)

    @pytest.mark.parametrize('full_covariance', [False, True])
    def test_complex_gaussian_sampling(self, full_covariance):
        from pyvoicebox.v_randvec import v_randvec

        np.random.seed(43)
        mean = np.array([[.5, -2.]])
        covariance = np.array([[2., .4], [.4, .5]]) if full_covariance else np.diag([2., .5])
        specified = covariance[:, :, None] if full_covariance else np.diag(covariance)[None, :]
        x, components = v_randvec(40000, mean, specified, [1.], mode='c')
        assert np.iscomplexobj(x)
        np.testing.assert_array_equal(components, np.zeros(len(x), dtype=int))
        np.testing.assert_allclose(x.mean(axis=0), mean[0], atol=.02)
        np.testing.assert_allclose(np.cov(x.real, rowvar=False), covariance / 2, atol=.025)
        np.testing.assert_allclose(np.cov(x.imag, rowvar=False), covariance / 2, atol=.025)


# ============================================================
# v_gaussmixd
# ============================================================
class TestGaussmixd:
    def test_conditional(self):
        from pyvoicebox.v_gaussmixd import v_gaussmixd
        # Simple 2D Gaussian, condition on first dim
        m = np.array([[0.0, 0.0]])
        v = np.array([[1.0, 1.0]])
        w = np.array([1.0])
        y = np.array([[1.0]])
        mz, vz = v_gaussmixd(y, m, v, w, b=np.array([1]))
        # For diagonal cov, conditioning doesn't affect other dimensions
        np.testing.assert_allclose(mz.ravel(), [0.0], atol=1e-10)


# ============================================================
# v_lognmpdf
# ============================================================
class TestLognmpdf:
    def test_1d(self):
        from pyvoicebox.v_lognmpdf import v_lognmpdf
        # For a 1D lognormal with natural mean m and variance v
        x = np.array([[0.5], [1.0], [2.0], [5.0]])
        m_val = np.array([1.0])
        v_val = np.array([0.5])
        p = v_lognmpdf(x, m_val, v_val)
        # All should be non-negative
        assert np.all(p >= 0)
        # PDF at 0 should be 0 (or near it)
        p_zero = v_lognmpdf(np.array([[0.0]]), m_val, v_val)
        np.testing.assert_allclose(p_zero, [0.0], atol=1e-15)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
