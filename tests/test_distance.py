"""Tests for distance measures and miscellaneous functions."""

import os
import numpy as np
import scipy.io as sio
import pytest

REF_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pyvoicebox-test', 'ref_data')


def load_ref(name):
    return sio.loadmat(os.path.join(REF_DIR, name), squeeze_me=True)


# ============================================================
# v_modsym
# ============================================================
class TestModsym:
    def test_angle_wrap(self):
        from pyvoicebox.v_modsym import v_modsym
        ref = load_ref('ref_modsym.mat')
        z, k = v_modsym(ref['x1'], -2 * np.pi)
        np.testing.assert_allclose(z, ref['z1'], rtol=1e-10)
        np.testing.assert_array_equal(k, ref['k1'])

    def test_default_modulus(self):
        from pyvoicebox.v_modsym import v_modsym
        ref = load_ref('ref_modsym.mat')
        z, k = v_modsym(2.3, 1)
        np.testing.assert_allclose(z, ref['z2'], rtol=1e-10)

    def test_with_reference(self):
        from pyvoicebox.v_modsym import v_modsym
        ref = load_ref('ref_modsym.mat')
        z, k = v_modsym(np.array([1.1, 2.5, 3.8]), 2, 1)
        np.testing.assert_allclose(z, ref['z3'], rtol=1e-10)


# ============================================================
# v_soundspeed
# ============================================================
class TestSoundspeed:
    def test_default(self):
        from pyvoicebox.v_soundspeed import v_soundspeed
        ref = load_ref('ref_soundspeed.mat')
        v, d, z = v_soundspeed(20)
        np.testing.assert_allclose(v, ref['v1'], rtol=1e-10)
        np.testing.assert_allclose(d, ref['d1'], rtol=1e-10)
        np.testing.assert_allclose(z, ref['z1s'], rtol=1e-10)

    def test_25deg(self):
        from pyvoicebox.v_soundspeed import v_soundspeed
        ref = load_ref('ref_soundspeed.mat')
        v, d, z = v_soundspeed(25, 1.0)
        np.testing.assert_allclose(v, ref['v2'], rtol=1e-10)
        np.testing.assert_allclose(d, ref['d2'], rtol=1e-10)
        np.testing.assert_allclose(z, ref['z2s'], rtol=1e-10)

    def test_0deg(self):
        from pyvoicebox.v_soundspeed import v_soundspeed
        ref = load_ref('ref_soundspeed.mat')
        v, d, z = v_soundspeed(0)
        np.testing.assert_allclose(v, ref['v3'], rtol=1e-10)
        np.testing.assert_allclose(d, ref['d3'], rtol=1e-10)
        np.testing.assert_allclose(z, ref['z3s'], rtol=1e-10)


# ============================================================
# v_mintrace
# ============================================================
class TestMintrace:
    def test_basic(self):
        from pyvoicebox.v_mintrace import v_mintrace
        ref = load_ref('ref_mintrace.mat')
        p = v_mintrace(ref['x_mt'])
        # MATLAB is 1-based, Python is 0-based
        p_matlab = ref['p_mt'].astype(int)
        np.testing.assert_array_equal(p, p_matlab - 1)


# ============================================================
# v_distchar
# ============================================================
class TestDistchar:
    def test_diagonal(self):
        from pyvoicebox.v_distchar import v_distchar
        ref = load_ref('ref_distchar.mat')
        d = v_distchar(ref['ar1'], ref['ar2'], 'd')
        np.testing.assert_allclose(d, ref['d_char_d'], rtol=1e-10)

    def test_cross(self):
        from pyvoicebox.v_distchar import v_distchar
        ref = load_ref('ref_distchar.mat')
        d = v_distchar(ref['ar1'][:2, :], ref['ar2'], 'x')
        np.testing.assert_allclose(d, ref['d_char_x'], rtol=1e-10)


# ============================================================
# v_distisar
# ============================================================
class TestDistisar:
    def test_diagonal(self):
        from pyvoicebox.v_distisar import v_distisar
        ref = load_ref('ref_distisar.mat')
        d = v_distisar(ref['ar1'], ref['ar2'], 'd')
        np.testing.assert_allclose(d, ref['d_isar_d'], rtol=1e-10)

    def test_cross(self):
        from pyvoicebox.v_distisar import v_distisar
        ref = load_ref('ref_distisar.mat')
        d = v_distisar(ref['ar1'][:2, :], ref['ar2'], 'x')
        np.testing.assert_allclose(d, ref['d_isar_x'], rtol=1e-10)


# ============================================================
# v_distitar
# ============================================================
class TestDistitar:
    def test_diagonal(self):
        from pyvoicebox.v_distitar import v_distitar
        ref = load_ref('ref_distitar.mat')
        d = v_distitar(ref['ar1'], ref['ar2'], 'd')
        np.testing.assert_allclose(d, ref['d_itar_d'], rtol=1e-10)

    def test_cross(self):
        from pyvoicebox.v_distitar import v_distitar
        ref = load_ref('ref_distitar.mat')
        d = v_distitar(ref['ar1'][:2, :], ref['ar2'], 'x')
        np.testing.assert_allclose(d, ref['d_itar_x'], rtol=1e-10)

    def test_exp_mode(self):
        from pyvoicebox.v_distitar import v_distitar
        ref = load_ref('ref_distitar.mat')
        d = v_distitar(ref['ar1'], ref['ar2'], 'e')
        np.testing.assert_allclose(d, ref['d_itar_e'], rtol=1e-10)


# ============================================================
# v_distchpf
# ============================================================
class TestDistchpf:
    def test_diagonal(self):
        from pyvoicebox.v_distchpf import v_distchpf
        ref = load_ref('ref_distchpf.mat')
        d = v_distchpf(ref['pf1'], ref['pf2'], 'd')
        np.testing.assert_allclose(d, ref['d_chpf_d'], rtol=1e-10)

    def test_cross(self):
        from pyvoicebox.v_distchpf import v_distchpf
        ref = load_ref('ref_distchpf.mat')
        d = v_distchpf(ref['pf1'][:2, :], ref['pf2'], 'x')
        np.testing.assert_allclose(d, ref['d_chpf_x'], rtol=1e-10)


# ============================================================
# v_distispf
# ============================================================
class TestDistispf:
    def test_diagonal(self):
        from pyvoicebox.v_distispf import v_distispf
        ref = load_ref('ref_distispf.mat')
        d = v_distispf(ref['pf1'], ref['pf2'], 'd')
        np.testing.assert_allclose(d, ref['d_ispf_d'], rtol=1e-10)

    def test_cross(self):
        from pyvoicebox.v_distispf import v_distispf
        ref = load_ref('ref_distispf.mat')
        d = v_distispf(ref['pf1'][:2, :], ref['pf2'], 'x')
        np.testing.assert_allclose(d, ref['d_ispf_x'], rtol=1e-10)


# ============================================================
# v_distitpf
# ============================================================
class TestDistitpf:
    def test_diagonal(self):
        from pyvoicebox.v_distitpf import v_distitpf
        ref = load_ref('ref_distitpf.mat')
        d = v_distitpf(ref['pf1'], ref['pf2'], 'd')
        np.testing.assert_allclose(d, ref['d_itpf_d'], rtol=1e-10)

    def test_cross(self):
        from pyvoicebox.v_distitpf import v_distitpf
        ref = load_ref('ref_distitpf.mat')
        d = v_distitpf(ref['pf1'][:2, :], ref['pf2'], 'x')
        np.testing.assert_allclose(d, ref['d_itpf_x'], rtol=1e-10)


# ============================================================
# v_qrdotmult
# ============================================================
class TestQrdotmult:
    def test_single(self):
        from pyvoicebox.v_qrdotmult import v_qrdotmult
        ref = load_ref('ref_qrdotmult.mat')
        q = v_qrdotmult(ref['q1'], ref['q2'])
        np.testing.assert_allclose(q, ref['qm'], rtol=1e-10)

    def test_batch(self):
        from pyvoicebox.v_qrdotmult import v_qrdotmult
        ref = load_ref('ref_qrdotmult.mat')
        q = v_qrdotmult(ref['q1b'], ref['q2b'])
        np.testing.assert_allclose(q, ref['qmb'], rtol=1e-10)


# ============================================================
# v_qrdotdiv
# ============================================================
class TestQrdotdiv:
    def test_divide(self):
        from pyvoicebox.v_qrdotdiv import v_qrdotdiv
        ref = load_ref('ref_qrdotdiv.mat')
        q = v_qrdotdiv(ref['q1'], ref['q2'])
        np.testing.assert_allclose(q, ref['qd1'], rtol=1e-10)

    def test_inverse(self):
        from pyvoicebox.v_qrdotdiv import v_qrdotdiv
        ref = load_ref('ref_qrdotdiv.mat')
        q = v_qrdotdiv(ref['q1'])
        np.testing.assert_allclose(q, ref['qd_inv'], rtol=1e-10)


# ============================================================
# v_qrdivide
# ============================================================
class TestQrdivide:
    def test_unit_quaternions(self):
        from pyvoicebox.v_qrdivide import v_qrdivide
        ref = load_ref('ref_qrdivide.mat')
        q = v_qrdivide(ref['qr1'], ref['qr2'])
        np.testing.assert_allclose(q, ref['qdiv'], rtol=1e-10)

    def test_general(self):
        from pyvoicebox.v_qrdivide import v_qrdivide
        ref = load_ref('ref_qrdivide.mat')
        q = v_qrdivide(ref['q1'], ref['q2'])
        np.testing.assert_allclose(q, ref['qdiv2'], rtol=1e-10)

    def test_inverse(self):
        from pyvoicebox.v_qrdivide import v_qrdivide
        ref = load_ref('ref_qrdivide.mat')
        q = v_qrdivide(ref['q1'])
        np.testing.assert_allclose(q, ref['qinv'], rtol=1e-10)


# ============================================================
# v_potsband
# ============================================================
class TestPotsband:
    def test_8k(self):
        from pyvoicebox.v_potsband import v_potsband
        ref = load_ref('ref_potsband.mat')
        b, a = v_potsband(8000)
        np.testing.assert_allclose(b, ref['b_pots'], rtol=1e-6, atol=1e-14)
        np.testing.assert_allclose(a, ref['a_pots'], rtol=1e-6, atol=1e-14)

    def test_16k(self):
        from pyvoicebox.v_potsband import v_potsband
        ref = load_ref('ref_potsband.mat')
        b, a = v_potsband(16000)
        np.testing.assert_allclose(b, ref['b_pots16'], rtol=1e-6, atol=1e-14)
        np.testing.assert_allclose(a, ref['a_pots16'], rtol=1e-6, atol=1e-14)


# ============================================================
# v_ewgrpdel
# ============================================================
class TestEwgrpdel:
    def test_impulses(self):
        from pyvoicebox.v_ewgrpdel import v_ewgrpdel
        ref = load_ref('ref_ewgrpdel.mat')
        y, m = v_ewgrpdel(ref['x_ew'], 21)
        np.testing.assert_allclose(y, ref['y_ew'], rtol=1e-10)
        assert m == ref['m_ew']


# ============================================================
# v_quadpeak - 1D
# ============================================================
class TestQuadpeak:
    def test_1d(self):
        from pyvoicebox.v_quadpeak import v_quadpeak
        ref = load_ref('ref_quadpeak.mat')
        v, x, t, m = v_quadpeak(ref['z_1d'])
        np.testing.assert_allclose(v, ref['v_qp'], rtol=1e-10)
        np.testing.assert_allclose(x, ref['x_qp'], rtol=1e-10)
        assert t == ref['t_qp']

    def test_2d(self):
        from pyvoicebox.v_quadpeak import v_quadpeak
        ref = load_ref('ref_quadpeak2d.mat')
        v, x, t, m = v_quadpeak(ref['z_2d'])
        np.testing.assert_allclose(v, ref['v_qp2'], rtol=1e-10)
        np.testing.assert_allclose(x, ref['x_qp2'], rtol=1e-10)
        assert t == ref['t_qp2']


# ============================================================
# v_hypergeom1f1
# ============================================================
class TestHypergeom1f1:
    def test_small_z(self):
        from pyvoicebox.v_hypergeom1f1 import v_hypergeom1f1
        ref = load_ref('ref_hypergeom1f1.mat')
        h, _ = v_hypergeom1f1(1, 2, 0.5)
        np.testing.assert_allclose(h, ref['h1'], rtol=1e-6)

    def test_negative_z(self):
        from pyvoicebox.v_hypergeom1f1 import v_hypergeom1f1
        ref = load_ref('ref_hypergeom1f1.mat')
        h, _ = v_hypergeom1f1(0.5, 1.5, -3)
        np.testing.assert_allclose(h, ref['h2'], rtol=1e-6)

    def test_array(self):
        from pyvoicebox.v_hypergeom1f1 import v_hypergeom1f1
        ref = load_ref('ref_hypergeom1f1.mat')
        h, _ = v_hypergeom1f1(2, 3, np.array([0.1, 0.5, 1.0, 5.0, -5.0]))
        np.testing.assert_allclose(h, ref['h3'], rtol=1e-6)


# ============================================================
# v_txalign
# ============================================================
class TestTxalign:
    def test_basic(self):
        from pyvoicebox.v_txalign import v_txalign
        ref = load_ref('ref_txalign.mat')
        kx, ky, nxy, mxy, sxy = v_txalign(
            ref['x_ta'].astype(float),
            ref['y_ta'].astype(float),
            1.1
        )
        np.testing.assert_array_equal(kx, ref['kx_ta'].astype(int))
        np.testing.assert_array_equal(ky, ref['ky_ta'].astype(int))
        assert nxy == int(ref['nxy_ta'])
        np.testing.assert_allclose(mxy, ref['mxy_ta'], rtol=1e-10)
        np.testing.assert_allclose(sxy, ref['sxy_ta'], rtol=1e-10)


# ============================================================
# Utility function tests (no Octave reference needed)
# ============================================================
class TestHostipinfo:
    def test_hostname(self):
        from pyvoicebox.v_hostipinfo import v_hostipinfo
        name = v_hostipinfo('h')
        assert isinstance(name, str)
        assert len(name) > 0


class TestUnixwhich:
    def test_python(self):
        from pyvoicebox.v_unixwhich import v_unixwhich
        result = v_unixwhich('python3')
        # python3 should be found on most systems
        if result is not None:
            assert 'python' in result.lower()

    def test_not_found(self):
        from pyvoicebox.v_unixwhich import v_unixwhich
        result = v_unixwhich('nonexistent_program_xyz123')
        assert result is None


class TestWinenvar:
    def test_path(self):
        from pyvoicebox.v_winenvar import v_winenvar
        result = v_winenvar('PATH')
        assert result is not None


class TestStubs:
    def test_voicebox_update(self):
        from pyvoicebox.v_voicebox_update import v_voicebox_update
        with pytest.raises(NotImplementedError):
            v_voicebox_update()

    def test_m2htmlpwd(self):
        from pyvoicebox.v_m2htmlpwd import v_m2htmlpwd
        with pytest.raises(NotImplementedError):
            v_m2htmlpwd()

    def test_sigma(self):
        from pyvoicebox.v_sigma import v_sigma
        with pytest.raises(NotImplementedError):
            v_sigma(np.zeros(100), 8000)

    def test_spendred(self):
        from pyvoicebox.v_spendred import v_spendred
        with pytest.raises(NotImplementedError):
            v_spendred()

    def test_sapisynth(self):
        from pyvoicebox.v_sapisynth import v_sapisynth
        with pytest.raises(NotImplementedError):
            v_sapisynth()

    def test_gammabank(self):
        from pyvoicebox.v_gammabank import v_gammabank
        with pytest.raises(NotImplementedError):
            v_gammabank()

    def test_sphrharm(self):
        from pyvoicebox.v_sphrharm import v_sphrharm
        with pytest.raises(NotImplementedError):
            v_sphrharm()

    def test_paramsetch(self):
        from pyvoicebox.v_paramsetch import v_paramsetch
        with pytest.raises(NotImplementedError):
            v_paramsetch()

    def test_psycest(self):
        from pyvoicebox.v_psycest import v_psycest
        with pytest.raises(NotImplementedError):
            v_psycest()

    def test_psycestu(self):
        from pyvoicebox.v_psycestu import v_psycestu
        with pytest.raises(NotImplementedError):
            v_psycestu()

    def test_psycdigit(self):
        from pyvoicebox.v_psycdigit import v_psycdigit
        with pytest.raises(NotImplementedError):
            v_psycdigit()
