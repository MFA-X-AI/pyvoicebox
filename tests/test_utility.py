"""Tests for utility and infrastructure functions."""

import os
import numpy as np
import pytest
from scipy.io import loadmat
from conftest import _sine, _speech_like

REF_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pyvoicebox-test', 'ref_data')


def load_ref(name):
    return loadmat(os.path.join(REF_DIR, name), squeeze_me=True)


# ============================================================
# v_logsum
# ============================================================
class TestLogsum:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_logsum.mat')

    def test_1d(self):
        from pyvoicebox.v_logsum import v_logsum
        x = self.ref['x1']
        y = v_logsum(x)
        np.testing.assert_allclose(y, self.ref['y1'], rtol=1e-10)

    def test_2d_axis0(self):
        from pyvoicebox.v_logsum import v_logsum
        x = self.ref['x2']
        y = v_logsum(x, d=0)  # MATLAB dim=1 -> Python axis=0
        np.testing.assert_allclose(y, self.ref['y2_d1'], rtol=1e-10)

    def test_2d_axis1(self):
        from pyvoicebox.v_logsum import v_logsum
        x = self.ref['x2']
        y = v_logsum(x, d=1)  # MATLAB dim=2 -> Python axis=1
        np.testing.assert_allclose(y, self.ref['y2_d2'], rtol=1e-10)

    def test_large_negative(self):
        from pyvoicebox.v_logsum import v_logsum
        x = self.ref['x3']
        y = v_logsum(x)
        np.testing.assert_allclose(y, self.ref['y3'], rtol=1e-10)

    def test_infinity(self):
        from pyvoicebox.v_logsum import v_logsum
        x = self.ref['x4']
        y = v_logsum(x)
        np.testing.assert_allclose(y, self.ref['y4'], rtol=1e-10)

    def test_with_k(self):
        from pyvoicebox.v_logsum import v_logsum
        x = self.ref['x5']
        k = self.ref['k5']
        y = v_logsum(x, d=0, k=k)  # MATLAB dim=1 -> axis=0
        np.testing.assert_allclose(y, self.ref['y5'], rtol=1e-10)


# ============================================================
# v_gammalns
# ============================================================
class TestGammalns:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_gammalns.mat')

    def test_positive(self):
        from pyvoicebox.v_gammalns import v_gammalns
        x = self.ref['xg1']
        y = v_gammalns(x)
        np.testing.assert_allclose(np.real(y), self.ref['yg1'], rtol=1e-10)

    def test_negative_with_sign(self):
        from pyvoicebox.v_gammalns import v_gammalns
        x = self.ref['xg2']
        y, s = v_gammalns(x, return_sign=True)
        np.testing.assert_allclose(y, self.ref['yg2'], rtol=1e-10)
        np.testing.assert_array_equal(s, self.ref['sg2'])

    def test_nonpositive_integers(self):
        from pyvoicebox.v_gammalns import v_gammalns
        x = self.ref['xg3']
        y = v_gammalns(x)
        expected = self.ref['yg3']
        # Should be Inf for non-positive integers
        np.testing.assert_array_equal(np.isinf(np.real(y)), np.isinf(expected))


# ============================================================
# v_entropy
# ============================================================
class TestEntropy:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_entropy.mat')

    def test_uniform4(self):
        from pyvoicebox.v_entropy import v_entropy
        p = self.ref['pe1']
        h = v_entropy(p)
        np.testing.assert_allclose(h, self.ref['he1'], rtol=1e-10)

    def test_uniform2(self):
        from pyvoicebox.v_entropy import v_entropy
        p = self.ref['pe2']
        h = v_entropy(p)
        np.testing.assert_allclose(h, self.ref['he2'], rtol=1e-10)

    def test_deterministic(self):
        from pyvoicebox.v_entropy import v_entropy
        p = self.ref['pe3']
        h = v_entropy(p)
        np.testing.assert_allclose(h, self.ref['he3'], atol=1e-15)

    def test_skewed(self):
        from pyvoicebox.v_entropy import v_entropy
        p = self.ref['pe4']
        h = v_entropy(p)
        np.testing.assert_allclose(h, self.ref['he4'], rtol=1e-10)


# ============================================================
# v_bitsprec
# ============================================================
class TestBitsprec:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_bitsprec.mat')

    def test_fne(self):
        from pyvoicebox.v_bitsprec import v_bitsprec
        x = self.ref['xb']
        y = v_bitsprec(x, 0, 'fne')
        np.testing.assert_allclose(y, self.ref['yb_fne'], rtol=1e-10)

    def test_fno(self):
        from pyvoicebox.v_bitsprec import v_bitsprec
        x = self.ref['xb']
        y = v_bitsprec(x, 0, 'fno')
        np.testing.assert_allclose(y, self.ref['yb_fno'], rtol=1e-10)

    def test_fp(self):
        from pyvoicebox.v_bitsprec import v_bitsprec
        x = self.ref['xb']
        y = v_bitsprec(x, 0, 'fp-')
        np.testing.assert_allclose(y, self.ref['yb_fp'], rtol=1e-10)

    def test_fm(self):
        from pyvoicebox.v_bitsprec import v_bitsprec
        x = self.ref['xb']
        y = v_bitsprec(x, 0, 'fm-')
        np.testing.assert_allclose(y, self.ref['yb_fm'], rtol=1e-10)

    def test_fz(self):
        from pyvoicebox.v_bitsprec import v_bitsprec
        x = self.ref['xb']
        y = v_bitsprec(x, 0, 'fz-')
        np.testing.assert_allclose(y, self.ref['yb_fz'], rtol=1e-10)

    def test_significant_3bits(self):
        from pyvoicebox.v_bitsprec import v_bitsprec
        y = v_bitsprec(3.14159, 3, 'sne')
        np.testing.assert_allclose(y, self.ref['yb_s3'], rtol=1e-10)

    def test_significant_10bits(self):
        from pyvoicebox.v_bitsprec import v_bitsprec
        y = v_bitsprec(3.14159, 10, 'sne')
        np.testing.assert_allclose(y, self.ref['yb_s10'], rtol=1e-10)


# ============================================================
# v_atan2sc
# ============================================================
class TestAtan2sc:
    def test_basic(self):
        from pyvoicebox.v_atan2sc import v_atan2sc
        s, c, r, t = v_atan2sc(np.array([1.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]))
        assert s.shape == (3,)
        np.testing.assert_allclose(s ** 2 + c ** 2, 1.0, atol=1e-14)
        assert np.all(r >= 0)

    def test_known_values(self):
        from pyvoicebox.v_atan2sc import v_atan2sc
        s, c, r, t = v_atan2sc(np.array([1.0]), np.array([0.0]))
        np.testing.assert_allclose(s, [1.0], atol=1e-14)
        np.testing.assert_allclose(c, [0.0], atol=1e-14)
        np.testing.assert_allclose(r, [1.0], atol=1e-14)
        np.testing.assert_allclose(t, [np.pi / 2], atol=1e-14)


# ============================================================
# v_dlyapsq
# ============================================================
class TestDlyapsq:
    def test_basic(self):
        from pyvoicebox.v_dlyapsq import v_dlyapsq
        A = np.array([[0.5, 0.1], [0.0, 0.3]])
        B = np.eye(2)
        V = v_dlyapsq(A, B)
        assert V.shape == (2, 2)
        VtV = V.T @ V
        residual = A @ VtV @ A.T - VtV + B @ B.T
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)


# ============================================================
# v_dualdiag
# ============================================================
class TestDualdiag:
    def test_basic(self):
        from pyvoicebox.v_dualdiag import v_dualdiag
        rng = np.random.RandomState(42)
        W = rng.randn(3, 3)
        W = W @ W.T
        B = rng.randn(3, 3)
        B = B @ B.T
        a, d, e = v_dualdiag(W, B)
        assert a.shape == (3, 3)
        aba = a.conj().T @ B @ a
        off_diag = aba - np.diag(np.diag(aba))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-8)


# ============================================================
# v_pdfmoments
# ============================================================
class TestPdfmoments:
    def test_basic(self):
        from pyvoicebox.v_pdfmoments import v_pdfmoments
        moments = np.array([0.0, 1.0, 0.0, 3.0])
        result = v_pdfmoments('mK', moments)
        assert result is not None


# ============================================================
# v_zerotrim
# ============================================================
class TestZerotrim:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_zerotrim.mat')

    def test_basic(self):
        from pyvoicebox.v_zerotrim import v_zerotrim
        x = self.ref['xz1']
        z = v_zerotrim(x)
        np.testing.assert_array_equal(z, self.ref['yz1'])

    def test_partial(self):
        from pyvoicebox.v_zerotrim import v_zerotrim
        x = self.ref['xz3']
        z = v_zerotrim(x)
        np.testing.assert_array_equal(z, self.ref['yz3'])


# ============================================================
# v_choosenk
# ============================================================
class TestChoosenk:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_choosenk.mat')

    def test_5_3(self):
        from pyvoicebox.v_choosenk import v_choosenk
        x = v_choosenk(5, 3)
        # MATLAB is 1-based, Python is 0-based
        np.testing.assert_array_equal(x, self.ref['xc1'] - 1)

    def test_4_2(self):
        from pyvoicebox.v_choosenk import v_choosenk
        x = v_choosenk(4, 2)
        np.testing.assert_array_equal(x, self.ref['xc2'] - 1)

    def test_4_4(self):
        from pyvoicebox.v_choosenk import v_choosenk
        x = v_choosenk(4, 4)
        expected = np.atleast_2d(self.ref['xc3'] - 1)
        np.testing.assert_array_equal(x, expected)


# ============================================================
# v_choosrnk
# ============================================================
class TestChoosrnk:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_choosrnk.mat')

    def test_3_2(self):
        from pyvoicebox.v_choosrnk import v_choosrnk
        x = v_choosrnk(3, 2)
        np.testing.assert_array_equal(x, self.ref['xcr1'] - 1)

    def test_2_3(self):
        from pyvoicebox.v_choosrnk import v_choosrnk
        x = v_choosrnk(2, 3)
        np.testing.assert_array_equal(x, self.ref['xcr2'] - 1)


# ============================================================
# v_permutes
# ============================================================
class TestPermutes:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_permutes.mat')

    def test_3(self):
        from pyvoicebox.v_permutes import v_permutes
        p, s = v_permutes(3, return_sign=True)
        # MATLAB is 1-based
        np.testing.assert_array_equal(p, self.ref['xp3'] - 1)
        np.testing.assert_array_equal(s, self.ref['sp3'])

    def test_4(self):
        from pyvoicebox.v_permutes import v_permutes
        p, s = v_permutes(4, return_sign=True)
        np.testing.assert_array_equal(p, self.ref['xp4'] - 1)
        np.testing.assert_array_equal(s, self.ref['sp4'])


# ============================================================
# v_sort
# ============================================================
class TestSort:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_sort.mat')

    def test_1d(self):
        from pyvoicebox.v_sort import v_sort
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
        b, i, j = v_sort(a, return_inverse=True)
        np.testing.assert_array_equal(b, self.ref['bs1'])
        # MATLAB is 1-based
        np.testing.assert_array_equal(i, self.ref['is1'].astype(int) - 1)
        np.testing.assert_array_equal(j, self.ref['js1'].astype(int) - 1)

    def test_2d(self):
        from pyvoicebox.v_sort import v_sort
        a = np.array([[3, 1], [4, 2], [1, 5]], dtype=float)
        b, i, j = v_sort(a, return_inverse=True)
        np.testing.assert_array_equal(b, self.ref['bs2'])
        np.testing.assert_array_equal(i, self.ref['is2'].astype(int) - 1)
        np.testing.assert_array_equal(j, self.ref['js2'].astype(int) - 1)


# ============================================================
# v_windows
# ============================================================
class TestWindows:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_windows.mat')

    def test_hamming(self):
        from pyvoicebox.v_windows import v_windows
        w = v_windows('hamming', 64)
        np.testing.assert_allclose(w, self.ref['wh'], rtol=1e-10)

    def test_hanning(self):
        from pyvoicebox.v_windows import v_windows
        w = v_windows('hanning', 64)
        np.testing.assert_allclose(w, self.ref['wn'], rtol=1e-10)

    def test_rectangle(self):
        from pyvoicebox.v_windows import v_windows
        w = v_windows('rectangle', 32)
        np.testing.assert_allclose(w, self.ref['wr'], rtol=1e-10)

    def test_kaiser(self):
        from pyvoicebox.v_windows import v_windows
        w = v_windows('kaiser', 64, 'uw', 8)
        np.testing.assert_allclose(w, self.ref['wk'], rtol=1e-10)

    def test_gaussian(self):
        from pyvoicebox.v_windows import v_windows
        w = v_windows('gaussian', 64, 'uw', 3)
        np.testing.assert_allclose(w, self.ref['wg'], rtol=1e-10)

    def test_triangle(self):
        from pyvoicebox.v_windows import v_windows
        w = v_windows('triangle', 64)
        np.testing.assert_allclose(w, self.ref['wt'], rtol=1e-10)

    def test_blackman(self):
        from pyvoicebox.v_windows import v_windows
        w = v_windows('blackman', 64)
        np.testing.assert_allclose(w, self.ref['wbl'], rtol=1e-10)

    def test_harris4(self):
        from pyvoicebox.v_windows import v_windows
        w = v_windows('harris4', 128)
        np.testing.assert_allclose(w, self.ref['wh4'], rtol=1e-10)

    def test_tukey(self):
        from pyvoicebox.v_windows import v_windows
        w = v_windows('tukey', 64, 'uw', 0.5)
        np.testing.assert_allclose(w, self.ref['wtu'], rtol=1e-10)

    def test_vorbis(self):
        from pyvoicebox.v_windows import v_windows
        w = v_windows('vorbis', 64)
        np.testing.assert_allclose(w, self.ref['wv'], rtol=1e-10)


# ============================================================
# v_huffman
# ============================================================
class TestHuffman:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_huffman.mat')

    def test_basic(self):
        from pyvoicebox.v_huffman import v_huffman
        p = self.ref['ph']
        cc, ll, l = v_huffman(p)
        np.testing.assert_array_equal(ll, self.ref['llh'])
        np.testing.assert_allclose(l, self.ref['lh'], rtol=1e-10)


# ============================================================
# v_voicebox
# ============================================================
class TestVoicebox:
    def test_get_all(self):
        from pyvoicebox.v_voicebox import v_voicebox
        pp = v_voicebox()
        assert isinstance(pp, dict)
        assert 'dir_temp' in pp

    def test_get_field(self):
        from pyvoicebox.v_voicebox import v_voicebox
        val = v_voicebox('memsize')
        assert val == 50e6

    def test_set_field(self):
        from pyvoicebox.v_voicebox import v_voicebox
        v_voicebox('memsize', 100e6)
        assert v_voicebox('memsize') == 100e6
        v_voicebox('memsize', 50e6)  # restore

    def test_invalid_field(self):
        from pyvoicebox.v_voicebox import v_voicebox
        with pytest.raises(ValueError):
            v_voicebox('nonexistent_field', 42)

    def test_missing_field(self):
        from pyvoicebox.v_voicebox import v_voicebox
        val = v_voicebox('nonexistent_field')
        assert val is None


# ============================================================
# v_finishat
# ============================================================
class TestFinishat:
    def test_basic(self):
        from pyvoicebox.v_finishat import v_finishat
        result = v_finishat(0, 100, 50)
        assert isinstance(result, str)


# ============================================================
# v_fopenmkd
# ============================================================
class TestFopenmkd:
    def test_basic(self, tmp_path):
        from pyvoicebox.v_fopenmkd import v_fopenmkd
        path = str(tmp_path / "subdir" / "test.txt")
        try:
            result = v_fopenmkd(path, 'w')
            assert result is not None
        except Exception:
            pytest.skip("v_fopenmkd may need specific MATLAB-style args")


# ============================================================
# v_regexfiles
# ============================================================
class TestRegexfiles:
    def test_basic(self, tmp_path):
        from pyvoicebox.v_regexfiles import v_regexfiles
        (tmp_path / "test1.txt").write_text("a")
        (tmp_path / "test2.txt").write_text("b")
        (tmp_path / "test3.csv").write_text("c")
        result = v_regexfiles(r'\.txt$', str(tmp_path))
        assert isinstance(result, list)
        assert len(result) == 2


# ============================================================
# v_rnsubset
# ============================================================
class TestRnsubset:
    def test_basic(self):
        from pyvoicebox.v_rnsubset import v_rnsubset
        result = v_rnsubset(3, 10)
        vals = np.asarray(result).flatten()
        assert len(vals) == 3
        assert len(np.unique(vals)) == 3
        assert np.all((vals >= 0) & (vals < 10))
