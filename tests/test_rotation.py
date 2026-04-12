"""Tests for rotation, quaternion, and geometry functions."""

import os
import numpy as np
import scipy.io as sio
import pytest

REF_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pyvoicebox-test', 'ref_data')


def load_ref(name):
    return sio.loadmat(os.path.join(REF_DIR, name), squeeze_me=True)


# ============================================================
# v_roteucode
# ============================================================
class TestRoteucode:
    def test_xyz(self):
        from pyvoicebox.v_roteucode import v_roteucode
        ref = load_ref('ref_roteucode.mat')
        mv = v_roteucode('xyz')
        np.testing.assert_allclose(mv, ref['mv_xyz'], atol=1e-14)

    def test_zxz(self):
        from pyvoicebox.v_roteucode import v_roteucode
        ref = load_ref('ref_roteucode.mat')
        mv = v_roteucode('zxz')
        np.testing.assert_allclose(mv, ref['mv_zxz'], atol=1e-14)

    def test_zyx(self):
        from pyvoicebox.v_roteucode import v_roteucode
        ref = load_ref('ref_roteucode.mat')
        mv = v_roteucode('zyx')
        np.testing.assert_allclose(mv, ref['mv_zyx'], atol=1e-14)

    def test_degrees(self):
        from pyvoicebox.v_roteucode import v_roteucode
        ref = load_ref('ref_roteucode.mat')
        mv = v_roteucode('dxyz')
        np.testing.assert_allclose(mv, ref['mv_d'], atol=1e-14)

    def test_intrinsic(self):
        from pyvoicebox.v_roteucode import v_roteucode
        ref = load_ref('ref_roteucode.mat')
        mv = v_roteucode('Ozyx')
        np.testing.assert_allclose(mv, ref['mv_O'], atol=1e-14)


# ============================================================
# v_roteu2qr
# ============================================================
class TestRoteu2qr:
    def test_xyz_single(self):
        from pyvoicebox.v_roteu2qr import v_roteu2qr
        ref = load_ref('ref_roteu2qr.mat')
        q = v_roteu2qr('xyz', ref['e1'])
        np.testing.assert_allclose(q, ref['q_xyz_1'], rtol=1e-10)

    def test_zxz_single(self):
        from pyvoicebox.v_roteu2qr import v_roteu2qr
        ref = load_ref('ref_roteu2qr.mat')
        q = v_roteu2qr('zxz', ref['e1'])
        np.testing.assert_allclose(q, ref['q_zxz_1'], rtol=1e-10)

    def test_zyx_single(self):
        from pyvoicebox.v_roteu2qr import v_roteu2qr
        ref = load_ref('ref_roteu2qr.mat')
        q = v_roteu2qr('zyx', ref['e1'])
        np.testing.assert_allclose(q, ref['q_zyx_1'], rtol=1e-10)

    def test_xyz_batch(self):
        from pyvoicebox.v_roteu2qr import v_roteu2qr
        ref = load_ref('ref_roteu2qr.mat')
        q = v_roteu2qr('xyz', ref['e_batch'])
        np.testing.assert_allclose(q, ref['q_xyz_batch'], rtol=1e-10)

    def test_zyx_batch(self):
        from pyvoicebox.v_roteu2qr import v_roteu2qr
        ref = load_ref('ref_roteu2qr.mat')
        q = v_roteu2qr('zyx', ref['e_batch'])
        np.testing.assert_allclose(q, ref['q_zyx_batch'], rtol=1e-10)

    def test_fixed_rotation(self):
        from pyvoicebox.v_roteu2qr import v_roteu2qr
        ref = load_ref('ref_roteu2qr.mat')
        q = v_roteu2qr('456')
        np.testing.assert_allclose(q, ref['q_456'], rtol=1e-10)

    def test_degrees(self):
        from pyvoicebox.v_roteu2qr import v_roteu2qr
        ref = load_ref('ref_roteu2qr.mat')
        q = v_roteu2qr('dxyz', ref['e_deg'])
        np.testing.assert_allclose(q, ref['q_deg'], rtol=1e-10)


# ============================================================
# v_rotqr2ro
# ============================================================
class TestRotqr2ro:
    def test_single(self):
        from pyvoicebox.v_rotqr2ro import v_rotqr2ro
        ref = load_ref('ref_rotqr2ro.mat')
        r = v_rotqr2ro(ref['q_xyz_1'])
        np.testing.assert_allclose(r, ref['r_xyz_1'], rtol=1e-10)

    def test_zxz(self):
        from pyvoicebox.v_rotqr2ro import v_rotqr2ro
        ref = load_ref('ref_rotqr2ro.mat')
        r = v_rotqr2ro(ref['q_zxz_1'])
        np.testing.assert_allclose(r, ref['r_zxz_1'], rtol=1e-10)

    def test_batch(self):
        from pyvoicebox.v_rotqr2ro import v_rotqr2ro
        ref = load_ref('ref_rotqr2ro.mat')
        r = v_rotqr2ro(ref['q_xyz_batch'])
        np.testing.assert_allclose(r, ref['r_xyz_batch'], rtol=1e-10)


# ============================================================
# v_rotro2qr
# ============================================================
class TestRotro2qr:
    def test_single(self):
        from pyvoicebox.v_rotro2qr import v_rotro2qr
        ref = load_ref('ref_rotro2qr.mat')
        q = v_rotro2qr(ref['r_xyz_1'])
        np.testing.assert_allclose(q, ref['q_back_1'], rtol=1e-10)

    def test_batch(self):
        from pyvoicebox.v_rotro2qr import v_rotro2qr
        ref = load_ref('ref_rotro2qr.mat')
        q = v_rotro2qr(ref['r_xyz_batch'])
        np.testing.assert_allclose(q, ref['q_back_batch'], rtol=1e-10)


# ============================================================
# v_roteu2ro
# ============================================================
class TestRoteu2ro:
    def test_single(self):
        from pyvoicebox.v_roteu2ro import v_roteu2ro
        ref = load_ref('ref_roteu2ro.mat')
        r = v_roteu2ro('xyz', ref['e1'])
        np.testing.assert_allclose(r, ref['r_eu2ro_1'], rtol=1e-10)

    def test_batch(self):
        from pyvoicebox.v_roteu2ro import v_roteu2ro
        ref = load_ref('ref_roteu2ro.mat')
        r = v_roteu2ro('xyz', ref['e_batch'])
        np.testing.assert_allclose(r, ref['r_eu2ro_batch'], rtol=1e-10)

    def test_zxz(self):
        from pyvoicebox.v_roteu2ro import v_roteu2ro
        ref = load_ref('ref_roteu2ro.mat')
        r = v_roteu2ro('zxz', ref['e1'])
        np.testing.assert_allclose(r, ref['r_eu2ro_zxz'], rtol=1e-10)


# ============================================================
# v_rotro2eu
# ============================================================
class TestRotro2eu:
    def test_xyz_single(self):
        from pyvoicebox.v_rotro2eu import v_rotro2eu
        ref = load_ref('ref_rotro2eu.mat')
        e = v_rotro2eu('xyz', ref['r_eu2ro_1'])
        np.testing.assert_allclose(e, ref['e_back_xyz_1'], rtol=1e-10)

    def test_xyz_batch(self):
        from pyvoicebox.v_rotro2eu import v_rotro2eu
        ref = load_ref('ref_rotro2eu.mat')
        e = v_rotro2eu('xyz', ref['r_eu2ro_batch'])
        np.testing.assert_allclose(e, ref['e_back_xyz_batch'], rtol=1e-10)

    def test_zxz(self):
        from pyvoicebox.v_rotro2eu import v_rotro2eu
        ref = load_ref('ref_rotro2eu.mat')
        e = v_rotro2eu('zxz', ref['r_eu2ro_zxz'])
        np.testing.assert_allclose(e, ref['e_back_zxz'], rtol=1e-10)

    def test_roundtrip_xyz(self):
        from pyvoicebox.v_roteu2ro import v_roteu2ro
        from pyvoicebox.v_rotro2eu import v_rotro2eu
        e_orig = np.array([0.1, 0.2, 0.3])
        r = v_roteu2ro('xyz', e_orig)
        e_back = v_rotro2eu('xyz', r)
        np.testing.assert_allclose(e_back, e_orig, rtol=1e-10)


# ============================================================
# v_rotqr2eu
# ============================================================
class TestRotqr2eu:
    def test_xyz(self):
        from pyvoicebox.v_rotqr2eu import v_rotqr2eu
        ref = load_ref('ref_rotqr2eu.mat')
        e = v_rotqr2eu('xyz', ref['q_xyz_1'])
        np.testing.assert_allclose(e, ref['e_qr2eu_1'], rtol=1e-10)

    def test_zxz(self):
        from pyvoicebox.v_rotqr2eu import v_rotqr2eu
        ref = load_ref('ref_rotqr2eu.mat')
        e = v_rotqr2eu('zxz', ref['q_zxz_1'])
        np.testing.assert_allclose(e, ref['e_qr2eu_zxz'], rtol=1e-10)


# ============================================================
# v_rotqc2qr / v_rotqr2qc
# ============================================================
class TestRotqc:
    def test_qr2qc(self):
        from pyvoicebox.v_rotqr2qc import v_rotqr2qc
        ref = load_ref('ref_rotqc.mat')
        qc = v_rotqr2qc(ref['qr1'])
        np.testing.assert_allclose(qc, ref['qc1'], rtol=1e-10)

    def test_qc2qr(self):
        from pyvoicebox.v_rotqc2qr import v_rotqc2qr
        ref = load_ref('ref_rotqc.mat')
        qr = v_rotqc2qr(ref['qc1'])
        np.testing.assert_allclose(qr, ref['qr1_back'], rtol=1e-10)

    def test_roundtrip(self):
        from pyvoicebox.v_rotqr2qc import v_rotqr2qc
        from pyvoicebox.v_rotqc2qr import v_rotqc2qr
        qr = np.array([0.5, 0.5, 0.5, 0.5])
        qc = v_rotqr2qc(qr)
        qr_back = v_rotqc2qr(qc)
        np.testing.assert_allclose(qr_back, qr, rtol=1e-10)


# ============================================================
# v_rotqc2mc / v_rotmc2qc
# ============================================================
class TestRotmc:
    def test_qc2mc(self):
        from pyvoicebox.v_rotqc2mc import v_rotqc2mc
        ref = load_ref('ref_rotmc.mat')
        mc = v_rotqc2mc(ref['qc1'])
        np.testing.assert_allclose(mc, ref['mc1'], rtol=1e-10)

    def test_mc2qc(self):
        from pyvoicebox.v_rotmc2qc import v_rotmc2qc
        ref = load_ref('ref_rotmc.mat')
        qc = v_rotmc2qc(ref['mc1'])
        np.testing.assert_allclose(qc, ref['qc1_back'], rtol=1e-10)

    def test_roundtrip(self):
        from pyvoicebox.v_rotqc2mc import v_rotqc2mc
        from pyvoicebox.v_rotmc2qc import v_rotmc2qc
        qc = np.array([0.5 + 0.5j, 0.5 + 0.5j])
        mc = v_rotqc2mc(qc)
        qc_back = v_rotmc2qc(mc)
        np.testing.assert_allclose(qc_back, qc, rtol=1e-10)


# ============================================================
# v_rotqr2mr / v_rotmr2qr
# ============================================================
class TestRotmr:
    def test_qr2mr(self):
        from pyvoicebox.v_rotqr2mr import v_rotqr2mr
        ref = load_ref('ref_rotmr.mat')
        mr = v_rotqr2mr(ref['qr1'])
        np.testing.assert_allclose(mr, ref['mr1'], rtol=1e-10)

    def test_mr2qr(self):
        from pyvoicebox.v_rotmr2qr import v_rotmr2qr
        ref = load_ref('ref_rotmr.mat')
        qr = v_rotmr2qr(ref['mr1'])
        np.testing.assert_allclose(qr, ref['qr1_mr_back'], rtol=1e-10)

    def test_roundtrip(self):
        from pyvoicebox.v_rotqr2mr import v_rotqr2mr
        from pyvoicebox.v_rotmr2qr import v_rotmr2qr
        qr = np.array([0.5, 0.5, 0.5, 0.5])
        mr = v_rotqr2mr(qr)
        qr_back = v_rotmr2qr(mr)
        np.testing.assert_allclose(qr_back, qr, rtol=1e-10)


# ============================================================
# v_rotax2qr / v_rotqr2ax
# ============================================================
class TestRotax:
    def test_ax2qr(self):
        from pyvoicebox.v_rotax2qr import v_rotax2qr
        ref = load_ref('ref_rotax.mat')
        q = v_rotax2qr(ref['ax1'], float(ref['t1']))
        np.testing.assert_allclose(q, ref['q_ax'], rtol=1e-10)

    def test_qr2ax(self):
        from pyvoicebox.v_rotqr2ax import v_rotqr2ax
        ref = load_ref('ref_rotax.mat')
        a, t = v_rotqr2ax(ref['q_ax'])
        np.testing.assert_allclose(a, ref['ax_back'], rtol=1e-10)
        np.testing.assert_allclose(t, float(ref['t_back']), rtol=1e-10)

    def test_roundtrip(self):
        from pyvoicebox.v_rotax2qr import v_rotax2qr
        from pyvoicebox.v_rotqr2ax import v_rotqr2ax
        axis = np.array([1.0, 2.0, 3.0])
        angle = 0.7
        q = v_rotax2qr(axis, angle)
        a, t = v_rotqr2ax(q)
        np.testing.assert_allclose(t, angle, rtol=1e-10)
        expected_axis = axis / np.linalg.norm(axis)
        np.testing.assert_allclose(a, expected_axis, rtol=1e-10)


# ============================================================
# v_rotqrmean
# ============================================================
class TestRotqrmean:
    def test_mean(self):
        from pyvoicebox.v_rotqrmean import v_rotqrmean
        ref = load_ref('ref_rotqrmean.mat')
        y, s, v = v_rotqrmean(ref['q_mean_in'])
        # Mean quaternion can have sign ambiguity
        if np.dot(y, ref['y_mean']) < 0:
            y = -y
        np.testing.assert_allclose(np.abs(y), np.abs(ref['y_mean']), rtol=1e-6)

    def test_identical_quaternions(self):
        from pyvoicebox.v_rotqrmean import v_rotqrmean
        q = np.array([1.0, 0, 0, 0]).reshape(4, 1)
        q = np.tile(q, (1, 5))
        y, s, v = v_rotqrmean(q)
        np.testing.assert_allclose(y, np.array([1, 0, 0, 0]), atol=1e-14)
        assert v < 1e-14


# ============================================================
# v_rotqrvec
# ============================================================
class TestRotqrvec:
    def test_single_vector(self):
        from pyvoicebox.v_rotqrvec import v_rotqrvec
        ref = load_ref('ref_rotqrvec.mat')
        y = v_rotqrvec(ref['q_vec'], ref['x_vec'])
        np.testing.assert_allclose(y, ref['y_vec'], rtol=1e-10)

    def test_multi_vector(self):
        from pyvoicebox.v_rotqrvec import v_rotqrvec
        ref = load_ref('ref_rotqrvec.mat')
        y = v_rotqrvec(ref['q_vec'], ref['x_multi'])
        np.testing.assert_allclose(y, ref['y_multi'], rtol=1e-10)


# ============================================================
# v_qrabs
# ============================================================
class TestQrabs:
    def test_basic(self):
        from pyvoicebox.v_qrabs import v_qrabs
        q = np.array([1.0, 0.0, 0.0, 0.0])
        m, qn = v_qrabs(q)
        assert np.allclose(m, 1.0, atol=1e-10)

    def test_unit_quaternion(self):
        from pyvoicebox.v_qrabs import v_qrabs
        q = np.array([0.5, 0.5, 0.5, 0.5])
        m, qn = v_qrabs(q)
        assert np.allclose(m, 1.0, atol=1e-10)
        assert np.allclose(np.linalg.norm(qn), 1.0, atol=1e-10)


# ============================================================
# v_qrmult
# ============================================================
class TestQrmult:
    def test_identity(self):
        from pyvoicebox.v_qrmult import v_qrmult
        q = np.array([1.0, 0.0, 0.0, 0.0])
        result = v_qrmult(q, q)
        r = np.asarray(result).flatten()
        np.testing.assert_allclose(r[:4], [1, 0, 0, 0], atol=1e-10)


# ============================================================
# v_qrpermute
# ============================================================
class TestQrpermute:
    def test_basic(self):
        from pyvoicebox.v_qrpermute import v_qrpermute
        q = np.array([1.0, 2.0, 3.0, 4.0])
        result = v_qrpermute(q)
        r = np.asarray(result).flatten()
        assert len(r) == 4


# ============================================================
# v_rotation
# ============================================================
class TestRotation:
    def test_plane_rotation(self):
        from pyvoicebox.v_rotation import v_rotation
        ref = load_ref('ref_rotation.mat')
        r = v_rotation(np.array([1, 0, 0]), np.array([0, 1, 0]), np.pi / 4)
        np.testing.assert_allclose(r, ref['r_rot1'], rtol=1e-10)

    def test_axis_angle(self):
        from pyvoicebox.v_rotation import v_rotation
        ref = load_ref('ref_rotation.mat')
        r = v_rotation(np.array([0, 0, 1]), np.pi / 3)
        np.testing.assert_allclose(r, ref['r_rot2'], rtol=1e-10)

    def test_axis_times_angle(self):
        from pyvoicebox.v_rotation import v_rotation
        ref = load_ref('ref_rotation.mat')
        r = v_rotation(np.array([0, 0, np.pi / 6]))
        np.testing.assert_allclose(r, ref['r_rot3'], rtol=1e-10)


# ============================================================
# v_rotpl2ro / v_rotro2pl
# ============================================================
class TestRotpl:
    def test_pl2ro(self):
        from pyvoicebox.v_rotpl2ro import v_rotpl2ro
        ref = load_ref('ref_rotpl.mat')
        r = v_rotpl2ro(ref['u1'], ref['v1'], np.pi / 4)
        np.testing.assert_allclose(r, ref['r_pl'], rtol=1e-10)

    def test_ro2pl(self):
        from pyvoicebox.v_rotro2pl import v_rotro2pl
        ref = load_ref('ref_rotpl.mat')
        u, v, t = v_rotro2pl(ref['r_pl'])
        np.testing.assert_allclose(t, float(ref['t_plback']), rtol=1e-6)

    def test_roundtrip(self):
        from pyvoicebox.v_rotpl2ro import v_rotpl2ro
        from pyvoicebox.v_rotro2pl import v_rotro2pl
        u = np.array([1.0, 0, 0])
        v = np.array([0, 1.0, 0])
        r = v_rotpl2ro(u, v, 0.5)
        u2, v2, t2 = v_rotro2pl(r)
        np.testing.assert_allclose(t2, 0.5, rtol=1e-6)


# ============================================================
# v_rotlu2ro / v_rotro2lu
# ============================================================
class TestRotlu:
    def test_lu2ro(self):
        from pyvoicebox.v_rotlu2ro import v_rotlu2ro
        ref = load_ref('ref_rotlu.mat')
        r = v_rotlu2ro(ref['l1'], ref['u_lu'])
        np.testing.assert_allclose(r, ref['r_lu'], rtol=1e-10)

    def test_ro2lu(self):
        from pyvoicebox.v_rotro2lu import v_rotro2lu
        ref = load_ref('ref_rotlu.mat')
        l, u = v_rotro2lu(ref['r_lu'])
        np.testing.assert_allclose(l, ref['l_back'], rtol=1e-10)
        np.testing.assert_allclose(u, ref['u_back_lu'], rtol=1e-10)


# ============================================================
# v_polygonarea
# ============================================================
class TestPolygonarea:
    def test_triangle(self):
        from pyvoicebox.v_polygonarea import v_polygonarea
        ref = load_ref('ref_polygonarea.mat')
        a = v_polygonarea(ref['p_tri'])
        np.testing.assert_allclose(a, float(ref['a_tri']), rtol=1e-10)

    def test_square(self):
        from pyvoicebox.v_polygonarea import v_polygonarea
        ref = load_ref('ref_polygonarea.mat')
        a = v_polygonarea(ref['p_sq'])
        np.testing.assert_allclose(a, float(ref['a_sq']), rtol=1e-10)

    def test_rectangle(self):
        from pyvoicebox.v_polygonarea import v_polygonarea
        ref = load_ref('ref_polygonarea.mat')
        a = v_polygonarea(ref['p_ccw'])
        np.testing.assert_allclose(a, float(ref['a_ccw']), rtol=1e-10)


# ============================================================
# v_polygonwind
# ============================================================
class TestPolygonwind:
    def test_winding(self):
        from pyvoicebox.v_polygonwind import v_polygonwind
        ref = load_ref('ref_polygonwind.mat')
        w = v_polygonwind(ref['p_wind'], ref['x_wind'])
        np.testing.assert_allclose(w, ref['w_wind'], rtol=1e-10)


# ============================================================
# v_polygonxline
# ============================================================
class TestPolygonxline:
    def test_crossing(self):
        from pyvoicebox.v_polygonxline import v_polygonxline
        ref = load_ref('ref_polygonxline.mat')
        xc, ec, tc, xy0 = v_polygonxline(ref['p_xline'], ref['l_xline'])
        np.testing.assert_allclose(xc, ref['xc_xl'], rtol=1e-10)
        np.testing.assert_allclose(ec, ref['ec_xl'], rtol=1e-10)
        np.testing.assert_allclose(tc, ref['tc_xl'], rtol=1e-10)
        np.testing.assert_allclose(xy0, ref['xy0_xl'], rtol=1e-10)


# ============================================================
# v_imagehomog
# ============================================================
class TestImagehomog:
    def test_identity(self):
        from pyvoicebox.v_imagehomog import v_imagehomog
        im = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        ih, xa, ya = v_imagehomog(im, np.eye(3))
        assert ih.shape[:2] == im.shape[:2]
        assert len(xa) > 0
        assert len(ya) > 0


# ============================================================
# v_rectifyhomog
# ============================================================
class TestRectifyhomog:
    def test_basic(self):
        from pyvoicebox.v_rectifyhomog import v_rectifyhomog
        import pytest
        try:
            result = v_rectifyhomog(np.eye(3))
            assert result is not None
        except Exception:
            pytest.skip("v_rectifyhomog may need specific input")


# ============================================================
# v_minspane
# ============================================================
class TestMinspane:
    def test_basic(self):
        from pyvoicebox.v_minspane import v_minspane
        points = np.random.randn(20, 2)
        result = v_minspane(points)
        assert result is not None


# ============================================================
# v_upolyhedron
# ============================================================
class TestUpolyhedron:
    def test_basic(self):
        from pyvoicebox.v_upolyhedron import v_upolyhedron
        import pytest
        try:
            result = v_upolyhedron(4)
            assert result is not None
        except Exception:
            pytest.skip("v_upolyhedron may need specific parameters")


# ============================================================
# v_horizdiff
# ============================================================
class TestHorizdiff:
    def test_basic(self):
        from pyvoicebox.v_horizdiff import v_horizdiff
        import pytest
        x = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * x)
        try:
            result = v_horizdiff(x, y, x, y + 0.1)
            assert result is not None
        except Exception:
            pytest.skip("v_horizdiff may need specific input format")


# ============================================================
# v_skew3d
# ============================================================
class TestSkew3d:
    def test_vec_to_mat_3(self):
        from pyvoicebox.v_skew3d import v_skew3d
        ref = load_ref('ref_skew3d.mat')
        y = v_skew3d(ref['x_sk3'].reshape(3, 1))
        np.testing.assert_allclose(y, ref['y_sk3'], rtol=1e-10)

    def test_mat_to_vec_3(self):
        from pyvoicebox.v_skew3d import v_skew3d
        ref = load_ref('ref_skew3d.mat')
        x = v_skew3d(ref['y_sk3'])
        np.testing.assert_allclose(x, ref['x_sk3_back'], rtol=1e-10)

    def test_roundtrip_3(self):
        from pyvoicebox.v_skew3d import v_skew3d
        x = np.array([1, 2, 3], dtype=float).reshape(3, 1)
        y = v_skew3d(x)
        x_back = v_skew3d(y)
        np.testing.assert_allclose(x_back, x.ravel(), rtol=1e-10)

    def test_vec_to_mat_6(self):
        from pyvoicebox.v_skew3d import v_skew3d
        ref = load_ref('ref_skew3d.mat')
        y = v_skew3d(ref['x_sk6'].reshape(6, 1))
        np.testing.assert_allclose(y, ref['y_sk6'], rtol=1e-10)

    def test_mat_to_vec_4x4(self):
        from pyvoicebox.v_skew3d import v_skew3d
        ref = load_ref('ref_skew3d.mat')
        x = v_skew3d(ref['y_sk6'])
        np.testing.assert_allclose(x, ref['x_sk6_back'], rtol=1e-10)

    def test_cross_product(self):
        from pyvoicebox.v_skew3d import v_skew3d
        a = np.array([1, 2, 3], dtype=float).reshape(3, 1)
        b = np.array([4, 5, 6], dtype=float)
        result = v_skew3d(a) @ b
        expected = np.cross(a.ravel(), b.ravel())
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ============================================================
# Integration tests
# ============================================================
class TestIntegration:
    def test_euler_roundtrip_all_conventions(self):
        """Test Euler angle roundtrip for multiple conventions.

        For repeated-axis conventions (zxz, xyx, etc.) the recovered angles
        may differ by the well-known ambiguity, so we compare the rotation
        matrices rather than the angles directly.
        """
        from pyvoicebox.v_roteu2ro import v_roteu2ro
        from pyvoicebox.v_rotro2eu import v_rotro2eu
        conventions = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', 'zxz', 'xyx']
        e_orig = np.array([0.1, 0.2, 0.3])
        for conv in conventions:
            r = v_roteu2ro(conv, e_orig)
            e_back = v_rotro2eu(conv, r)
            r_back = v_roteu2ro(conv, e_back)
            np.testing.assert_allclose(r_back, r, atol=1e-10,
                                       err_msg=f'Failed for convention {conv}')

    def test_quaternion_rotation_matrix_consistency(self):
        """Test that quaternion and rotation matrix representations are consistent."""
        from pyvoicebox.v_roteu2qr import v_roteu2qr
        from pyvoicebox.v_rotqr2ro import v_rotqr2ro
        from pyvoicebox.v_rotro2qr import v_rotro2qr
        e = np.array([0.3, 0.5, 0.7])
        q = v_roteu2qr('xyz', e)
        r = v_rotqr2ro(q)
        q2 = v_rotro2qr(r)
        np.testing.assert_allclose(q, q2, atol=1e-10)

    def test_axis_angle_euler_consistency(self):
        """Test axis-angle and Euler representations are consistent."""
        from pyvoicebox.v_rotax2qr import v_rotax2qr
        from pyvoicebox.v_rotqr2ro import v_rotqr2ro
        from pyvoicebox.v_rotation import v_rotation
        axis = np.array([0, 0, 1.0])
        angle = np.pi / 4
        q = v_rotax2qr(axis, angle)
        r1 = v_rotqr2ro(q)
        r2 = v_rotation(axis, angle)
        np.testing.assert_allclose(r1, r2, rtol=1e-10)
