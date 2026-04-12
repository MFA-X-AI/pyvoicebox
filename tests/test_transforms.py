"""Tests for FFT, DCT, and transform functions."""

import os
import numpy as np
import pytest
from scipy.io import loadmat

REF_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pyvoicebox-test', 'ref_data')


def load_ref(name):
    return loadmat(os.path.join(REF_DIR, name), squeeze_me=True)


# ============================================================
# v_rfft
# ============================================================
class TestRfft:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_rfft.mat')

    def test_1d_even(self):
        from pyvoicebox.v_rfft import v_rfft
        x = self.ref['x_rfft_1d_even']
        y = v_rfft(x)
        np.testing.assert_allclose(y, self.ref['y_rfft_1d_even'], rtol=1e-10)

    def test_1d_odd(self):
        from pyvoicebox.v_rfft import v_rfft
        x = self.ref['x_rfft_1d_odd']
        y = v_rfft(x)
        np.testing.assert_allclose(y, self.ref['y_rfft_1d_odd'], rtol=1e-10)

    def test_1d_n16(self):
        from pyvoicebox.v_rfft import v_rfft
        x = self.ref['x_rfft_1d_even']
        y = v_rfft(x, n=16)
        np.testing.assert_allclose(y, self.ref['y_rfft_1d_n16'], rtol=1e-10)

    def test_1d_n4(self):
        from pyvoicebox.v_rfft import v_rfft
        x = self.ref['x_rfft_1d_even']
        y = v_rfft(x, n=4)
        np.testing.assert_allclose(y, self.ref['y_rfft_1d_n4'], rtol=1e-10)

    def test_2d_default(self):
        from pyvoicebox.v_rfft import v_rfft
        x = self.ref['x_rfft_2d']
        y = v_rfft(x)
        np.testing.assert_allclose(y, self.ref['y_rfft_2d'], rtol=1e-10)

    def test_2d_dim2(self):
        from pyvoicebox.v_rfft import v_rfft
        x = self.ref['x_rfft_2d']
        # MATLAB dim=2 -> Python axis=1
        y = v_rfft(x, d=1)
        np.testing.assert_allclose(y, self.ref['y_rfft_2d_d2'], rtol=1e-10)

    def test_scalar(self):
        from pyvoicebox.v_rfft import v_rfft
        y = v_rfft(np.array(42.0))
        np.testing.assert_allclose(y, 42.0, rtol=1e-10)


# ============================================================
# v_irfft
# ============================================================
class TestIrfft:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_irfft.mat')

    def test_even_output(self):
        from pyvoicebox.v_irfft import v_irfft
        y = self.ref['y_irfft_in_even']
        x = v_irfft(y, n=8)
        np.testing.assert_allclose(x, self.ref['x_irfft_even'], rtol=1e-10)

    def test_odd_output(self):
        from pyvoicebox.v_irfft import v_irfft
        y = self.ref['y_irfft_in_odd']
        x = v_irfft(y, n=7)
        np.testing.assert_allclose(x, self.ref['x_irfft_odd'], rtol=1e-10)

    def test_default_n(self):
        from pyvoicebox.v_irfft import v_irfft
        y = self.ref['y_irfft_in_even']
        x = v_irfft(y)
        np.testing.assert_allclose(x, self.ref['x_irfft_default'], rtol=1e-10)

    def test_2d(self):
        from pyvoicebox.v_irfft import v_irfft
        y = self.ref['y_irfft_2d_in']
        x = v_irfft(y, n=6)
        np.testing.assert_allclose(x, self.ref['x_irfft_2d'], rtol=1e-10)

    def test_roundtrip(self):
        from pyvoicebox.v_rfft import v_rfft
        from pyvoicebox.v_irfft import v_irfft
        x_orig = self.ref['x_roundtrip']
        y = v_rfft(x_orig, n=8)
        x_back = v_irfft(y, n=8)
        np.testing.assert_allclose(x_back, self.ref['x_roundtrip_back'], rtol=1e-10)


# ============================================================
# v_rsfft
# ============================================================
class TestRsfft:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_rsfft.mat')

    def test_1d_default(self):
        from pyvoicebox.v_rsfft import v_rsfft
        x = self.ref['x_rsfft_1d']
        y = v_rsfft(x)
        np.testing.assert_allclose(y, self.ref['y_rsfft_1d'], rtol=1e-10)

    def test_1d_n8(self):
        from pyvoicebox.v_rsfft import v_rsfft
        x = self.ref['x_rsfft_1d']
        y = v_rsfft(x, n=8)
        np.testing.assert_allclose(y, self.ref['y_rsfft_1d_n8'], rtol=1e-10)

    def test_1d_n9(self):
        from pyvoicebox.v_rsfft import v_rsfft
        x = self.ref['x_rsfft_1d']
        y = v_rsfft(x, n=9)
        np.testing.assert_allclose(y, self.ref['y_rsfft_1d_n9'], rtol=1e-10)

    def test_2d(self):
        from pyvoicebox.v_rsfft import v_rsfft
        x = self.ref['x_rsfft_2d']
        y = v_rsfft(x)
        np.testing.assert_allclose(y, self.ref['y_rsfft_2d'], rtol=1e-10)

    def test_self_inverse(self):
        from pyvoicebox.v_rsfft import v_rsfft
        x = self.ref['x_rsfft_si']
        n = int(self.ref['n_rsfft_si'])
        y = v_rsfft(x, n=n)
        x_back = v_rsfft(y, n=n) / n
        np.testing.assert_allclose(x_back, self.ref['x_rsfft_si_back'], rtol=1e-10)


# ============================================================
# v_rdct
# ============================================================
class TestRdct:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_rdct.mat')

    def test_1d_default(self):
        from pyvoicebox.v_rdct import v_rdct
        x = self.ref['x_rdct_1d']
        y = v_rdct(x)
        np.testing.assert_allclose(y, self.ref['y_rdct_1d'], rtol=1e-10)

    def test_1d_n4(self):
        from pyvoicebox.v_rdct import v_rdct
        x = self.ref['x_rdct_1d']
        y = v_rdct(x, n=4)
        np.testing.assert_allclose(y, self.ref['y_rdct_1d_n4'], rtol=1e-10)

    def test_1d_n16(self):
        from pyvoicebox.v_rdct import v_rdct
        x = self.ref['x_rdct_1d']
        y = v_rdct(x, n=16)
        np.testing.assert_allclose(y, self.ref['y_rdct_1d_n16'], rtol=1e-10)

    def test_1d_custom_ab(self):
        from pyvoicebox.v_rdct import v_rdct
        x = self.ref['x_rdct_1d']
        y = v_rdct(x, n=8, a=1.0, b=1.0)
        np.testing.assert_allclose(y, self.ref['y_rdct_1d_ab'], rtol=1e-10)

    def test_2d(self):
        from pyvoicebox.v_rdct import v_rdct
        x = self.ref['x_rdct_2d']
        y = v_rdct(x)
        np.testing.assert_allclose(y, self.ref['y_rdct_2d'], rtol=1e-10)

    def test_odd_length(self):
        from pyvoicebox.v_rdct import v_rdct
        x = self.ref['x_rdct_odd']
        y = v_rdct(x)
        np.testing.assert_allclose(y, self.ref['y_rdct_odd'], rtol=1e-10, atol=1e-14)


# ============================================================
# v_irdct
# ============================================================
class TestIrdct:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_irdct.mat')

    def test_1d_default(self):
        from pyvoicebox.v_irdct import v_irdct
        y = self.ref['y_irdct_1d']
        x = v_irdct(y)
        np.testing.assert_allclose(x, self.ref['x_irdct_1d'], rtol=1e-10)

    def test_roundtrip_even(self):
        from pyvoicebox.v_rdct import v_rdct
        from pyvoicebox.v_irdct import v_irdct
        x_orig = self.ref['x_irdct_rt_even']
        y = v_rdct(x_orig, n=8)
        x_back = v_irdct(y, n=8)
        np.testing.assert_allclose(x_back, self.ref['x_irdct_rt_even_back'], rtol=1e-10)

    def test_roundtrip_odd(self):
        from pyvoicebox.v_rdct import v_rdct
        from pyvoicebox.v_irdct import v_irdct
        x_orig = self.ref['x_irdct_rt_odd']
        y = v_rdct(x_orig, n=7)
        x_back = v_irdct(y, n=7)
        np.testing.assert_allclose(x_back, self.ref['x_irdct_rt_odd_back'], rtol=1e-10)

    def test_custom_ab(self):
        from pyvoicebox.v_rdct import v_rdct
        from pyvoicebox.v_irdct import v_irdct
        x_orig = self.ref['x_irdct_ab']
        y = v_rdct(x_orig, n=4, a=1.0, b=1.0)
        x_back = v_irdct(y, n=4, a=1.0, b=1.0)
        np.testing.assert_allclose(x_back, self.ref['x_irdct_ab_back'], rtol=1e-10)

    def test_2d(self):
        from pyvoicebox.v_rdct import v_rdct
        from pyvoicebox.v_irdct import v_irdct
        x_orig = self.ref['x_irdct_2d']
        y = v_rdct(x_orig)
        x_back = v_irdct(y)
        np.testing.assert_allclose(x_back, self.ref['x_irdct_2d_back'], rtol=1e-10)


# ============================================================
# v_rhartley
# ============================================================
class TestRhartley:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_rhartley.mat')

    def test_1d_default(self):
        from pyvoicebox.v_rhartley import v_rhartley
        x = self.ref['x_hart_1d']
        y = v_rhartley(x)
        np.testing.assert_allclose(y, self.ref['y_hart_1d'], rtol=1e-10)

    def test_1d_n16(self):
        from pyvoicebox.v_rhartley import v_rhartley
        x = self.ref['x_hart_1d']
        y = v_rhartley(x, n=16)
        np.testing.assert_allclose(y, self.ref['y_hart_1d_n16'], rtol=1e-10)

    def test_1d_n4(self):
        from pyvoicebox.v_rhartley import v_rhartley
        x = self.ref['x_hart_1d']
        y = v_rhartley(x, n=4)
        np.testing.assert_allclose(y, self.ref['y_hart_1d_n4'], rtol=1e-10)

    def test_2d(self):
        from pyvoicebox.v_rhartley import v_rhartley
        x = self.ref['x_hart_2d']
        y = v_rhartley(x)
        np.testing.assert_allclose(y, self.ref['y_hart_2d'], rtol=1e-10)

    def test_self_inverse(self):
        from pyvoicebox.v_rhartley import v_rhartley
        x = self.ref['x_hart_1d']
        y = v_rhartley(x, n=8)
        x_back = v_rhartley(y, n=8) / 8
        np.testing.assert_allclose(x_back, self.ref['x_hart_si_back'], rtol=1e-10)


# ============================================================
# v_zoomfft
# ============================================================
class TestZoomfft:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_zoomfft.mat')

    def test_equivalent_to_fft(self):
        from pyvoicebox.v_zoomfft import v_zoomfft
        x = self.ref['x_zoom_1d']
        y, f = v_zoomfft(x, n=8, m=8, s=0)
        np.testing.assert_allclose(y, self.ref['y_zoom_fft'], rtol=1e-10)
        np.testing.assert_allclose(f, self.ref['f_zoom_fft'], rtol=1e-10)

    def test_zoom_range(self):
        from pyvoicebox.v_zoomfft import v_zoomfft
        x = self.ref['x_zoom_1d']
        y, f = v_zoomfft(x, n=16, m=4, s=2)
        np.testing.assert_allclose(y, self.ref['y_zoom_range'], rtol=1e-10)
        np.testing.assert_allclose(f, self.ref['f_zoom_range'], rtol=1e-10)

    def test_chirp_path(self):
        from pyvoicebox.v_zoomfft import v_zoomfft
        x = self.ref['x_zoom_1d']
        y, f = v_zoomfft(x, n=10.5, m=6, s=1.5)
        np.testing.assert_allclose(y, self.ref['y_zoom_chirp'], rtol=1e-10)
        np.testing.assert_allclose(f, self.ref['f_zoom_chirp'], rtol=1e-10)

    def test_2d(self):
        from pyvoicebox.v_zoomfft import v_zoomfft
        x = self.ref['x_zoom_2d']
        y, f = v_zoomfft(x, n=4, m=4, s=0)
        np.testing.assert_allclose(y, self.ref['y_zoom_2d'], rtol=1e-10)
        np.testing.assert_allclose(f, self.ref['f_zoom_2d'], rtol=1e-10)


# ============================================================
# v_convfft
# ============================================================
class TestConvfft:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_convfft.mat')

    def test_filter_mode(self):
        from pyvoicebox.v_convfft import v_convfft
        x = self.ref['x_conv']
        h = self.ref['h_conv']
        z = v_convfft(x, h)
        np.testing.assert_allclose(z.ravel(), self.ref['z_conv_filter'], rtol=1e-10)

    def test_full_convolution(self):
        from pyvoicebox.v_convfft import v_convfft
        x = self.ref['x_conv']
        h = self.ref['h_conv']
        z = v_convfft(x, h, d=0, m='', h0=1, x1=1, x2=len(x) + len(h) - 1)
        np.testing.assert_allclose(z.ravel(), self.ref['z_conv_full'], rtol=1e-10)

    def test_correlation(self):
        from pyvoicebox.v_convfft import v_convfft
        x = self.ref['x_conv']
        h = self.ref['h_conv']
        z = v_convfft(x, h, d=0, m='x', h0=1, x1=1, x2=len(x))
        np.testing.assert_allclose(z.ravel(), self.ref['z_conv_xcorr'], rtol=1e-10)


# ============================================================
# v_frac2bin
# ============================================================
class TestFrac2bin:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_frac2bin.mat')

    def test_integer(self):
        from pyvoicebox.v_frac2bin import v_frac2bin
        s = v_frac2bin(np.array([5.0]))
        assert s[0] == str(self.ref['s_frac2bin_1'])

    def test_fractional(self):
        from pyvoicebox.v_frac2bin import v_frac2bin
        s = v_frac2bin(np.array([5.75]), n=1, m=4)
        assert s[0] == str(self.ref['s_frac2bin_2'])

    def test_vector(self):
        from pyvoicebox.v_frac2bin import v_frac2bin
        s = v_frac2bin(np.array([3.0, 5.0, 7.0]), n=4, m=0)
        ref = self.ref['s_frac2bin_3']
        for i in range(len(s)):
            assert s[i] == str(ref[i])

    def test_leading_spaces(self):
        from pyvoicebox.v_frac2bin import v_frac2bin
        s = v_frac2bin(np.array([1.0, 8.0]), n=-1, m=0)
        ref = self.ref['s_frac2bin_4']
        for i in range(len(s)):
            assert s[i] == str(ref[i])

    def test_truncation(self):
        from pyvoicebox.v_frac2bin import v_frac2bin
        s = v_frac2bin(np.array([5.75]), n=1, m=-4)
        assert s[0] == str(self.ref['s_frac2bin_5'])
