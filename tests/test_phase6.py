"""Tests for Phase 6: LPC (Linear Predictive Coding) functions."""

import os
import numpy as np
import pytest
from scipy.io import loadmat

REF_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pyvoicebox-test', 'ref_data')


def load_ref(name):
    return loadmat(os.path.join(REF_DIR, name), squeeze_me=True)


# ============================================================
# v_lpcauto
# ============================================================
class TestLpcauto:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcauto.mat')

    def test_single_frame(self):
        from pyvoicebox.v_lpcauto import v_lpcauto
        s = self.ref['s_auto']
        ar, e, k = v_lpcauto(s, 4)
        np.testing.assert_allclose(ar, self.ref['ar_auto'].reshape(ar.shape), rtol=1e-10)
        np.testing.assert_allclose(e, self.ref['e_auto'], rtol=1e-10)

    def test_multi_frame(self):
        from pyvoicebox.v_lpcauto import v_lpcauto
        s = self.ref['s_auto']
        ar, e, k = v_lpcauto(s, 4, np.array([[50, 50]]))
        np.testing.assert_allclose(ar, self.ref['ar_auto_f'], rtol=1e-10)
        np.testing.assert_allclose(e, self.ref['e_auto_f'], rtol=1e-10)


# ============================================================
# v_lpccovar
# ============================================================
class TestLpccovar:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpccovar.mat')

    def test_basic(self):
        from pyvoicebox.v_lpccovar import v_lpccovar
        s = self.ref['s_covar']
        ar, e, dc = v_lpccovar(s, 4)
        np.testing.assert_allclose(ar, self.ref['ar_covar'].reshape(ar.shape), rtol=1e-10)
        np.testing.assert_allclose(e, self.ref['e_covar'].reshape(e.shape), rtol=1e-10)


# ============================================================
# v_lpcbwexp
# ============================================================
class TestLpcbwexp:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcbwexp.mat')

    def test_basic(self):
        from pyvoicebox.v_lpcbwexp import v_lpcbwexp
        ar = self.ref['ar_bw']
        result = v_lpcbwexp(ar, 0.1)
        np.testing.assert_allclose(result.ravel(), self.ref['arx_bw'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcstable
# ============================================================
class TestLpcstable:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcstable.mat')

    def test_stability(self):
        from pyvoicebox.v_lpcstable import v_lpcstable
        ar = self.ref['ar_stable']
        m, a = v_lpcstable(ar)
        np.testing.assert_array_equal(m.astype(int), self.ref['m_stable'].astype(int))
        np.testing.assert_allclose(a, self.ref['a_stable'], rtol=1e-6)


# ============================================================
# v_lpcar2rf / v_lpcrf2ar roundtrip
# ============================================================
class TestAr2Rf:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcar2rf.mat')

    def test_ar2rf(self):
        from pyvoicebox.v_lpcar2rf import v_lpcar2rf
        ar = self.ref['ar_rf']
        rf = v_lpcar2rf(ar)
        np.testing.assert_allclose(rf, self.ref['rf_from_ar'], rtol=1e-10)

    def test_rf2ar(self):
        from pyvoicebox.v_lpcrf2ar import v_lpcrf2ar
        rf = self.ref['rf_from_ar']
        ar = v_lpcrf2ar(rf)
        np.testing.assert_allclose(ar, self.ref['ar_from_rf'], rtol=1e-10)

    def test_roundtrip(self):
        from pyvoicebox.v_lpcar2rf import v_lpcar2rf
        from pyvoicebox.v_lpcrf2ar import v_lpcrf2ar
        ar = self.ref['ar_rf']
        rf = v_lpcar2rf(ar)
        ar2 = v_lpcrf2ar(rf)
        np.testing.assert_allclose(ar2, ar, rtol=1e-10)


# ============================================================
# v_lpcar2cc / v_lpccc2ar roundtrip
# ============================================================
class TestAr2Cc:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcar2cc.mat')

    def test_ar2cc(self):
        from pyvoicebox.v_lpcar2cc import v_lpcar2cc
        ar = self.ref['ar_cc']
        cc, c0 = v_lpcar2cc(np.atleast_2d(ar))
        np.testing.assert_allclose(cc.ravel(), self.ref['cc_from_ar'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(c0.ravel(), np.atleast_1d(self.ref['c0_from_ar']).ravel(), rtol=1e-10)

    def test_ar2cc_extended(self):
        from pyvoicebox.v_lpcar2cc import v_lpcar2cc
        ar = self.ref['ar_cc']
        cc, _ = v_lpcar2cc(np.atleast_2d(ar), 6)
        np.testing.assert_allclose(cc.ravel(), self.ref['cc_from_ar6'].ravel(), rtol=1e-10)

    def test_cc2ar(self):
        from pyvoicebox.v_lpccc2ar import v_lpccc2ar
        cc = np.atleast_2d(self.ref['cc_from_ar'])
        ar = v_lpccc2ar(cc)
        np.testing.assert_allclose(ar.ravel(), self.ref['ar_from_cc'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcar2ff, v_lpcar2pf, v_lpcar2db
# ============================================================
class TestAr2Spec:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcar2spec.mat')

    def test_ar2ff(self):
        from pyvoicebox.v_lpcar2ff import v_lpcar2ff
        ar = self.ref['ar_spec']
        ff, f = v_lpcar2ff(np.atleast_2d(ar), 8)
        np.testing.assert_allclose(ff.ravel(), self.ref['ff_spec'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(f, self.ref['f_ff'], rtol=1e-10)

    def test_ar2pf(self):
        from pyvoicebox.v_lpcar2pf import v_lpcar2pf
        ar = self.ref['ar_spec']
        pf, f = v_lpcar2pf(np.atleast_2d(ar), 8)
        np.testing.assert_allclose(pf.ravel(), self.ref['pf_spec'].ravel(), rtol=1e-10)

    def test_ar2db(self):
        from pyvoicebox.v_lpcar2db import v_lpcar2db
        ar = self.ref['ar_spec']
        db, f = v_lpcar2db(np.atleast_2d(ar), 8)
        np.testing.assert_allclose(db.ravel(), self.ref['db_spec'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcar2im / v_lpcim2ar roundtrip
# ============================================================
class TestAr2Im:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcar2im.mat')

    def test_ar2im(self):
        from pyvoicebox.v_lpcar2im import v_lpcar2im
        ar = self.ref['ar_im']
        im = v_lpcar2im(np.atleast_2d(ar), 10)
        np.testing.assert_allclose(im.ravel(), self.ref['im_from_ar'].ravel(), rtol=1e-10)

    def test_im2ar(self):
        from pyvoicebox.v_lpcim2ar import v_lpcim2ar
        im = np.atleast_2d(self.ref['im_from_ar'])
        ar = v_lpcim2ar(im)
        np.testing.assert_allclose(ar.ravel(), self.ref['ar_from_im'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcar2rr / v_lpcrr2ar roundtrip
# ============================================================
class TestAr2Rr:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcar2rr.mat')

    def test_ar2rr(self):
        from pyvoicebox.v_lpcar2rr import v_lpcar2rr
        ar = self.ref['ar_rr']
        rr = v_lpcar2rr(np.atleast_2d(ar))
        np.testing.assert_allclose(rr.ravel(), self.ref['rr_from_ar'].ravel(), rtol=1e-10)

    def test_rr2ar(self):
        from pyvoicebox.v_lpcrr2ar import v_lpcrr2ar
        rr = np.atleast_2d(self.ref['rr_from_ar'])
        ar, e = v_lpcrr2ar(rr)
        np.testing.assert_allclose(ar.ravel(), self.ref['ar_from_rr'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(e.ravel(), np.atleast_1d(self.ref['e_from_rr']).ravel(), rtol=1e-10)


# ============================================================
# v_lpcar2ra
# ============================================================
class TestAr2Ra:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcar2ra.mat')

    def test_ar2ra(self):
        from pyvoicebox.v_lpcar2ra import v_lpcar2ra
        ar = self.ref['ar_ra']
        ra = v_lpcar2ra(np.atleast_2d(ar))
        np.testing.assert_allclose(ra.ravel(), self.ref['ra_from_ar'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcar2zz / v_lpczz2ar roundtrip
# ============================================================
class TestAr2Zz:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcar2zz.mat')

    def test_ar2zz(self):
        from pyvoicebox.v_lpcar2zz import v_lpcar2zz
        ar = self.ref['ar_zz']
        zz = v_lpcar2zz(np.atleast_2d(ar))
        # Sort by angle for comparison since root order may differ
        ref_zz = np.atleast_2d(self.ref['zz_from_ar'])
        zz_sorted = np.sort_complex(zz.ravel())
        ref_sorted = np.sort_complex(ref_zz.ravel())
        np.testing.assert_allclose(zz_sorted, ref_sorted, rtol=1e-6)

    def test_zz2ar(self):
        from pyvoicebox.v_lpczz2ar import v_lpczz2ar
        zz = np.atleast_2d(self.ref['zz_from_ar'])
        ar = v_lpczz2ar(zz)
        np.testing.assert_allclose(ar.ravel(), self.ref['ar_from_zz'].ravel(), rtol=1e-6)


# ============================================================
# v_lpcar2pp / v_lpcra2pp
# ============================================================
class TestAr2Pp:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcar2pp.mat')

    def test_ar2pp(self):
        from pyvoicebox.v_lpcar2pp import v_lpcar2pp
        ar = self.ref['ar_pp']
        pp = v_lpcar2pp(np.atleast_2d(ar))
        np.testing.assert_allclose(pp.ravel(), self.ref['pp_from_ar'].ravel(), rtol=1e-10)

    def test_ra2pp(self):
        from pyvoicebox.v_lpcra2pp import v_lpcra2pp
        ra = np.atleast_2d(self.ref['ra_pp'])
        pp = v_lpcra2pp(ra)
        np.testing.assert_allclose(pp.ravel(), self.ref['pp_from_ra'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcar2ls / v_lpcls2ar roundtrip
# ============================================================
class TestAr2Ls:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcar2ls.mat')

    def test_ar2ls(self):
        from pyvoicebox.v_lpcar2ls import v_lpcar2ls
        ar = self.ref['ar_ls']
        ls = v_lpcar2ls(np.atleast_2d(ar))
        np.testing.assert_allclose(ls.ravel(), self.ref['ls_from_ar'].ravel(), rtol=1e-6)

    def test_ls2ar(self):
        from pyvoicebox.v_lpcls2ar import v_lpcls2ar
        ls = np.atleast_2d(self.ref['ls_from_ar'])
        ar = v_lpcls2ar(ls)
        np.testing.assert_allclose(ar.ravel(), self.ref['ar_from_ls'].ravel(), rtol=1e-6)


# ============================================================
# v_lpcrf2* conversions
# ============================================================
class TestRfConversions:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcrf_conv.mat')

    def test_rf2aa(self):
        from pyvoicebox.v_lpcrf2aa import v_lpcrf2aa
        rf = self.ref['rf_test']
        aa = v_lpcrf2aa(np.atleast_2d(rf))
        np.testing.assert_allclose(aa.ravel(), self.ref['aa_from_rf'].ravel(), rtol=1e-10)

    def test_rf2ao(self):
        from pyvoicebox.v_lpcrf2ao import v_lpcrf2ao
        rf = self.ref['rf_test']
        ao = v_lpcrf2ao(rf)
        np.testing.assert_allclose(ao.ravel(), self.ref['ao_from_rf'].ravel(), rtol=1e-10)

    def test_rf2is(self):
        from pyvoicebox.v_lpcrf2is import v_lpcrf2is
        rf = self.ref['rf_test']
        is_coef = v_lpcrf2is(rf)
        np.testing.assert_allclose(is_coef.ravel(), self.ref['is_from_rf'].ravel(), rtol=1e-10)

    def test_rf2la(self):
        from pyvoicebox.v_lpcrf2la import v_lpcrf2la
        rf = self.ref['rf_test']
        la = v_lpcrf2la(np.atleast_2d(rf))
        np.testing.assert_allclose(la.ravel(), self.ref['la_from_rf'].ravel(), rtol=1e-10)

    def test_rf2lo(self):
        from pyvoicebox.v_lpcrf2lo import v_lpcrf2lo
        rf = self.ref['rf_test']
        lo = v_lpcrf2lo(rf)
        np.testing.assert_allclose(lo.ravel(), self.ref['lo_from_rf'].ravel(), rtol=1e-10)

    def test_rf2rr(self):
        from pyvoicebox.v_lpcrf2rr import v_lpcrf2rr
        rf = self.ref['rf_test']
        rr, ar = v_lpcrf2rr(np.atleast_2d(rf))
        np.testing.assert_allclose(rr.ravel(), self.ref['rr_from_rf'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(ar.ravel(), self.ref['ar_from_rf_rr'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcaa2* / v_lpcao2rf
# ============================================================
class TestAaConversions:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcaa_conv.mat')

    def test_aa2ao(self):
        from pyvoicebox.v_lpcaa2ao import v_lpcaa2ao
        aa = self.ref['aa_test']
        ao = v_lpcaa2ao(np.atleast_2d(aa))
        np.testing.assert_allclose(ao.ravel(), self.ref['ao_from_aa'].ravel(), rtol=1e-10)

    def test_aa2rf(self):
        from pyvoicebox.v_lpcaa2rf import v_lpcaa2rf
        aa = self.ref['aa_test']
        rf = v_lpcaa2rf(np.atleast_2d(aa))
        np.testing.assert_allclose(rf.ravel(), self.ref['rf_from_aa'].ravel(), rtol=1e-10)

    def test_ao2rf(self):
        from pyvoicebox.v_lpcao2rf import v_lpcao2rf
        ao = self.ref['ao_from_aa']
        rf = v_lpcao2rf(ao)
        np.testing.assert_allclose(rf.ravel(), self.ref['rf_from_ao'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcis2rf / v_lpcla2rf / v_lpclo2rf
# ============================================================
class TestInverseConversions:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpc_inv_conv.mat')

    def test_is2rf(self):
        from pyvoicebox.v_lpcis2rf import v_lpcis2rf
        is_coef = self.ref['is_from_rf']
        rf = v_lpcis2rf(is_coef)
        np.testing.assert_allclose(rf.ravel(), self.ref['rf_from_is'].ravel(), rtol=1e-10)

    def test_la2rf(self):
        from pyvoicebox.v_lpcla2rf import v_lpcla2rf
        la = self.ref['la_from_rf']
        rf = v_lpcla2rf(np.atleast_2d(la))
        np.testing.assert_allclose(rf.ravel(), self.ref['rf_from_la'].ravel(), rtol=1e-10)

    def test_lo2rf(self):
        from pyvoicebox.v_lpclo2rf import v_lpclo2rf
        lo = self.ref['lo_from_rf']
        rf = v_lpclo2rf(lo)
        np.testing.assert_allclose(rf.ravel(), self.ref['rf_from_lo'].ravel(), rtol=1e-10)


# ============================================================
# v_lpccc2cc, v_lpccc2db, v_lpccc2pf, v_lpccc2ff
# ============================================================
class TestCcConversions:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpccc_conv.mat')

    def test_cc2cc(self):
        from pyvoicebox.v_lpccc2cc import v_lpccc2cc
        cc = np.atleast_2d(self.ref['cc_test'])
        cc_ext = v_lpccc2cc(cc, 6)
        np.testing.assert_allclose(cc_ext.ravel(), self.ref['cc_ext'].ravel(), rtol=1e-10)

    def test_cc2db(self):
        from pyvoicebox.v_lpccc2db import v_lpccc2db
        cc = np.atleast_2d(self.ref['cc_test'])
        db, f = v_lpccc2db(cc, 8)
        np.testing.assert_allclose(db.ravel(), self.ref['db_from_cc'].ravel(), rtol=1e-10)

    def test_cc2pf(self):
        from pyvoicebox.v_lpccc2pf import v_lpccc2pf
        cc = np.atleast_2d(self.ref['cc_test'])
        pf, f = v_lpccc2pf(cc, 8)
        np.testing.assert_allclose(pf.ravel(), self.ref['pf_from_cc'].ravel(), rtol=1e-10)

    def test_cc2ff(self):
        from pyvoicebox.v_lpccc2ff import v_lpccc2ff
        cc = np.atleast_2d(self.ref['cc_test'])
        ff, f = v_lpccc2ff(cc, 8)
        np.testing.assert_allclose(ff.ravel(), self.ref['ff_from_cc'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcdb2pf / v_lpcff2pf
# ============================================================
class TestPfConversions:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpc_pf_conv.mat')

    def test_db2pf(self):
        from pyvoicebox.v_lpcdb2pf import v_lpcdb2pf
        db = self.ref['db_test']
        pf = v_lpcdb2pf(db)
        np.testing.assert_allclose(pf.ravel(), self.ref['pf_from_db'].ravel(), rtol=1e-10)

    def test_ff2pf(self):
        from pyvoicebox.v_lpcff2pf import v_lpcff2pf
        ff = self.ref['ff_test']
        pf = v_lpcff2pf(ff)
        np.testing.assert_allclose(pf.ravel(), self.ref['pf_from_ff'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcpf2rr / v_lpcpf2cc
# ============================================================
class TestPfToOthers:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcpf_conv.mat')

    def test_pf2rr(self):
        from pyvoicebox.v_lpcpf2rr import v_lpcpf2rr
        pf = np.atleast_2d(self.ref['pf_test'])
        rr = v_lpcpf2rr(pf, 3)
        np.testing.assert_allclose(rr.ravel(), self.ref['rr_from_pf'].ravel(), rtol=1e-10)

    def test_pf2cc(self):
        from pyvoicebox.v_lpcpf2cc import v_lpcpf2cc
        pf = np.atleast_2d(self.ref['pf_test'])
        cc, c0 = v_lpcpf2cc(pf, 3)
        np.testing.assert_allclose(cc.ravel(), self.ref['cc_from_pf'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(c0.ravel(), np.atleast_1d(self.ref['c0_from_pf']).ravel(), rtol=1e-10)


# ============================================================
# v_lpcrr2am / v_lpcar2am
# ============================================================
class TestAmConversions:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpc_am.mat')

    def test_rr2am(self):
        from pyvoicebox.v_lpcrr2am import v_lpcrr2am
        rr = np.atleast_2d(self.ref['rr_am'])
        am, em = v_lpcrr2am(rr)
        np.testing.assert_allclose(am.ravel(), self.ref['am_from_rr'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(em.ravel(), self.ref['em_from_rr'].ravel(), rtol=1e-10)

    def test_ar2am(self):
        from pyvoicebox.v_lpcar2am import v_lpcar2am
        ar = np.atleast_2d(self.ref['ar_am'])
        am, em = v_lpcar2am(ar)
        np.testing.assert_allclose(am.ravel(), self.ref['am_from_ar'].ravel(), rtol=1e-10)
        np.testing.assert_allclose(em.ravel(), self.ref['em_from_ar'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcss2zz / v_lpczz2ss
# ============================================================
class TestSsZz:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpc_sszz.mat')

    def test_ss2zz(self):
        from pyvoicebox.v_lpcss2zz import v_lpcss2zz
        ss = self.ref['ss_test']
        zz = v_lpcss2zz(np.atleast_2d(ss))
        np.testing.assert_allclose(zz.ravel(), self.ref['zz_from_ss'].ravel(), rtol=1e-10)

    def test_zz2ss(self):
        from pyvoicebox.v_lpczz2ss import v_lpczz2ss
        zz = self.ref['zz_from_ss']
        ss = v_lpczz2ss(zz)
        np.testing.assert_allclose(ss.ravel(), self.ref['ss_from_zz'].ravel(), rtol=1e-10)


# ============================================================
# v_lpczz2cc
# ============================================================
class TestZz2Cc:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpczz2cc.mat')

    def test_zz2cc(self):
        from pyvoicebox.v_lpczz2cc import v_lpczz2cc
        zz = np.atleast_2d(self.ref['zz_cc'])
        cc = v_lpczz2cc(zz, 5)
        np.testing.assert_allclose(cc.ravel(), self.ref['cc_from_zz'].ravel(), rtol=1e-10)


# ============================================================
# v_lpccw2zz / v_lpcpz2zz
# ============================================================
class TestCwPz:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpc_cwpz.mat')

    def test_cw2zz(self):
        from pyvoicebox.v_lpccw2zz import v_lpccw2zz
        cw = self.ref['cw_test']
        zz = v_lpccw2zz(cw)
        np.testing.assert_allclose(zz.ravel(), self.ref['zz_from_cw'].ravel(), rtol=1e-10)

    def test_pz2zz(self):
        from pyvoicebox.v_lpcpz2zz import v_lpcpz2zz
        pz = self.ref['pz_test']
        zz = v_lpcpz2zz(pz)
        np.testing.assert_allclose(zz.ravel(), self.ref['zz_from_pz'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcra2ar (Wilson spectral factorization)
# ============================================================
class TestRa2Ar:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcra2ar.mat')

    def test_ra2ar(self):
        from pyvoicebox.v_lpcra2ar import v_lpcra2ar
        ra = np.atleast_2d(self.ref['ra_for_wilson'])
        ar = v_lpcra2ar(ra)
        np.testing.assert_allclose(ar.ravel(), self.ref['ar_from_ra'].ravel(), rtol=1e-6)


# ============================================================
# v_lpcra2pf
# ============================================================
class TestRa2Pf:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcra2pf.mat')

    def test_ra2pf(self):
        from pyvoicebox.v_lpcra2pf import v_lpcra2pf
        ra = np.atleast_2d(self.ref['ra_pf'])
        pf = v_lpcra2pf(ra, 8)
        np.testing.assert_allclose(pf.ravel(), self.ref['pf_from_ra'].ravel(), rtol=1e-10)


# ============================================================
# v_lpcpp2cw
# ============================================================
class TestPp2Cw:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcpp2cw.mat')

    def test_pp2cw(self):
        from pyvoicebox.v_lpcpp2cw import v_lpcpp2cw
        pp = np.atleast_2d(self.ref['pp_test'])
        cw = v_lpcpp2cw(pp)
        # Sort by real part since root order may differ
        cw_sorted = np.sort_complex(cw.ravel())
        ref_sorted = np.sort_complex(self.ref['cw_from_pp'].ravel())
        np.testing.assert_allclose(cw_sorted, ref_sorted, rtol=1e-6)


# ============================================================
# v_lpcaa2dl / v_lpcdl2aa roundtrip
# ============================================================
class TestAaDl:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcaa2dl.mat')

    def test_aa2dl(self):
        from pyvoicebox.v_lpcaa2dl import v_lpcaa2dl
        aa = np.atleast_2d(self.ref['aa_dl'])
        dl = v_lpcaa2dl(aa)
        np.testing.assert_allclose(dl.ravel(), self.ref['dl_from_aa'].ravel(), rtol=1e-10)

    def test_dl2aa(self):
        from pyvoicebox.v_lpcdl2aa import v_lpcdl2aa
        dl = np.atleast_2d(self.ref['dl_from_aa'])
        aa = v_lpcdl2aa(dl)
        np.testing.assert_allclose(aa.ravel(), self.ref['aa_from_dl'].ravel(), rtol=1e-10)


# ============================================================
# v_rootstab
# ============================================================
class TestRootstab:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_rootstab.mat')

    def test_stable(self):
        from pyvoicebox.v_rootstab import v_rootstab
        p = self.ref['p_stable_poly']
        no, ni, nc = v_rootstab(p)
        assert no == int(self.ref['no_s'])
        assert ni == int(self.ref['ni_s'])
        assert nc == int(self.ref['nc_s'])

    def test_unstable(self):
        from pyvoicebox.v_rootstab import v_rootstab
        p = self.ref['p_unstable_poly']
        no, ni, nc = v_rootstab(p)
        assert no == int(self.ref['no_u'])
        assert ni == int(self.ref['ni_u'])
        assert nc == int(self.ref['nc_u'])


# ============================================================
# v_lpcfq2zz
# ============================================================
class TestFq2Zz:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lpcfq2zz.mat')

    def test_fq2zz(self):
        from pyvoicebox.v_lpcfq2zz import v_lpcfq2zz
        f = np.atleast_2d(self.ref['f_fq'])
        q = np.atleast_2d(self.ref['q_fq'])
        zz = v_lpcfq2zz(f, q)
        np.testing.assert_allclose(zz.ravel(), self.ref['zz_from_fq'].ravel(), rtol=1e-10)
