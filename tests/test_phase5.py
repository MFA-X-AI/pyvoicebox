"""Tests for Phase 5: Audio I/O & Codec functions."""

import os
import tempfile
import numpy as np
import pytest
from scipy.io import loadmat

REF_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pyvoicebox-test', 'ref_data')


def load_ref(name):
    return loadmat(os.path.join(REF_DIR, name), squeeze_me=True)


# ============================================================
# v_lin2pcmu / v_pcmu2lin  (Mu-law codec)
# ============================================================
class TestLin2Pcmu:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lin2pcmu.mat')

    def test_encode_s1(self):
        """Encode with scale factor 1."""
        from pyvoicebox.v_lin2pcmu import v_lin2pcmu
        x = self.ref['x_pcmu_test']
        expected = self.ref['pcmu_from_lin_s1']
        result = v_lin2pcmu(x, s=1)
        np.testing.assert_array_equal(result, expected)

    def test_encode_default(self):
        """Encode with default scale factor."""
        from pyvoicebox.v_lin2pcmu import v_lin2pcmu
        x = self.ref['x_pcmu_test'] / 4004.189931
        expected = self.ref['pcmu_from_lin_default']
        result = v_lin2pcmu(x)
        np.testing.assert_array_equal(result, expected)

    def test_roundtrip(self):
        """Encode then decode should match MATLAB roundtrip."""
        from pyvoicebox.v_lin2pcmu import v_lin2pcmu
        from pyvoicebox.v_pcmu2lin import v_pcmu2lin
        x = self.ref['x_pcmu_rt']
        encoded = v_lin2pcmu(x, s=1)
        np.testing.assert_array_equal(encoded, self.ref['pcmu_encoded'])
        decoded = v_pcmu2lin(encoded, s=1)
        np.testing.assert_allclose(decoded, self.ref['pcmu_decoded'], rtol=1e-10)


class TestPcmu2Lin:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lin2pcmu.mat')

    def test_decode_default(self):
        """Decode all 256 values with default scaling."""
        from pyvoicebox.v_pcmu2lin import v_pcmu2lin
        pcmu_all = self.ref['pcmu_all']
        expected = self.ref['lin_from_pcmu_default']
        result = v_pcmu2lin(pcmu_all)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_decode_s1(self):
        """Decode all 256 values with s=1."""
        from pyvoicebox.v_pcmu2lin import v_pcmu2lin
        pcmu_all = self.ref['pcmu_all']
        expected = self.ref['lin_from_pcmu_s1']
        result = v_pcmu2lin(pcmu_all, s=1)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_decode_s8031(self):
        """Decode all 256 values with s=8031."""
        from pyvoicebox.v_pcmu2lin import v_pcmu2lin
        pcmu_all = self.ref['pcmu_all']
        expected = self.ref['lin_from_pcmu_s8031']
        result = v_pcmu2lin(pcmu_all, s=8031)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_itu_sine(self):
        """ITU standard sine wave reference."""
        from pyvoicebox.v_pcmu2lin import v_pcmu2lin
        pcmu_sine = self.ref['pcmu_sine']
        expected = self.ref['lin_sine_pcmu']
        result = v_pcmu2lin(pcmu_sine)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ============================================================
# v_lin2pcma / v_pcma2lin  (A-law codec)
# ============================================================
class TestLin2Pcma:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lin2pcma.mat')

    def test_encode_s1(self):
        """Encode with scale factor 1 and default mask."""
        from pyvoicebox.v_lin2pcma import v_lin2pcma
        x = self.ref['x_pcma_test']
        expected = self.ref['pcma_from_lin_s1']
        result = v_lin2pcma(x, m=85, s=1)
        np.testing.assert_array_equal(result, expected)

    def test_encode_default(self):
        """Encode with default scale factor."""
        from pyvoicebox.v_lin2pcma import v_lin2pcma
        x = self.ref['x_pcma_test'] / 2017.396342
        expected = self.ref['pcma_from_lin_default']
        result = v_lin2pcma(x)
        np.testing.assert_array_equal(result, expected)

    def test_encode_no_mask(self):
        """Encode with no XOR mask."""
        from pyvoicebox.v_lin2pcma import v_lin2pcma
        x = self.ref['x_pcma_test']
        expected = self.ref['pcma_nomask']
        result = v_lin2pcma(x, m=0, s=1)
        np.testing.assert_array_equal(result, expected)

    def test_roundtrip(self):
        """Encode then decode should match MATLAB roundtrip."""
        from pyvoicebox.v_lin2pcma import v_lin2pcma
        from pyvoicebox.v_pcma2lin import v_pcma2lin
        x = self.ref['x_pcma_rt']
        encoded = v_lin2pcma(x, m=85, s=1)
        np.testing.assert_array_equal(encoded, self.ref['pcma_encoded'])
        decoded = v_pcma2lin(encoded, m=85, s=1)
        np.testing.assert_allclose(decoded, self.ref['pcma_decoded'], rtol=1e-10)


class TestPcma2Lin:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_lin2pcma.mat')

    def test_decode_default(self):
        """Decode all 256 values with default scaling."""
        from pyvoicebox.v_pcma2lin import v_pcma2lin
        pcma_all = self.ref['pcma_all']
        expected = self.ref['lin_from_pcma_default']
        result = v_pcma2lin(pcma_all)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_decode_s1(self):
        """Decode all 256 values with s=1."""
        from pyvoicebox.v_pcma2lin import v_pcma2lin
        pcma_all = self.ref['pcma_all']
        expected = self.ref['lin_from_pcma_s1']
        result = v_pcma2lin(pcma_all, m=85, s=1)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_decode_s4032(self):
        """Decode all 256 values with s=4032."""
        from pyvoicebox.v_pcma2lin import v_pcma2lin
        pcma_all = self.ref['pcma_all']
        expected = self.ref['lin_from_pcma_s4032']
        result = v_pcma2lin(pcma_all, m=85, s=4032)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_decode_no_mask(self):
        """Decode with no XOR mask."""
        from pyvoicebox.v_pcma2lin import v_pcma2lin
        pcma_all = self.ref['pcma_all']
        expected = self.ref['lin_from_pcma_nomask']
        result = v_pcma2lin(pcma_all, m=0, s=1)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_itu_sine(self):
        """ITU standard sine wave reference."""
        from pyvoicebox.v_pcma2lin import v_pcma2lin
        pcma_sine = self.ref['pcma_sine']
        expected = self.ref['lin_sine_pcma']
        result = v_pcma2lin(pcma_sine)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ============================================================
# v_readwav / v_writewav (WAV I/O)
# ============================================================
class TestWav:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_wav.mat')

    def test_read_mono(self):
        """Read mono 16-bit WAV file."""
        from pyvoicebox.v_readwav import v_readwav
        wav_file = os.path.join(REF_DIR, 'test_mono16.wav')
        y, fs = v_readwav(wav_file, 'p')
        expected = self.ref['wav_read_mono']
        if expected.ndim == 1:
            expected = expected.reshape(-1, 1)
        assert fs == int(self.ref['wav_fs_mono'])
        np.testing.assert_allclose(y, expected, atol=1e-4)

    def test_read_stereo(self):
        """Read stereo 16-bit WAV file."""
        from pyvoicebox.v_readwav import v_readwav
        wav_file = os.path.join(REF_DIR, 'test_stereo16.wav')
        y, fs = v_readwav(wav_file, 'p')
        expected = self.ref['wav_read_stereo']
        assert fs == int(self.ref['wav_fs_stereo'])
        np.testing.assert_allclose(y, expected, atol=1e-4)

    def test_read_partial(self):
        """Read with nmax and nskip."""
        from pyvoicebox.v_readwav import v_readwav
        wav_file = os.path.join(REF_DIR, 'test_mono16.wav')
        y, fs = v_readwav(wav_file, 'p', nmax=20, nskip=10)
        expected = self.ref['wav_read_partial']
        if expected.ndim == 1:
            expected = expected.reshape(-1, 1)
        assert y.shape[0] == 20
        np.testing.assert_allclose(y, expected, atol=1e-4)

    def test_write_read_roundtrip(self):
        """Write a WAV file and read it back."""
        from pyvoicebox.v_writewav import v_writewav
        from pyvoicebox.v_readwav import v_readwav
        rng = np.random.default_rng(42)
        # Use data in +-1 range so 'p' mode works well with 16-bit PCM
        data = rng.standard_normal((50, 2)) * 0.5
        data = np.clip(data, -1.0, 1.0)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmpfile = f.name
        try:
            v_writewav(data, 16000, tmpfile, 'p16')
            y, fs = v_readwav(tmpfile, 'p')
            assert fs == 16000
            assert y.shape == (50, 2)
            # 16-bit quantization means about 1/32768 error
            np.testing.assert_allclose(y, data, atol=1e-4)
        finally:
            os.unlink(tmpfile)

    def test_fs_preserved(self):
        """Sample rate is preserved through write/read."""
        from pyvoicebox.v_writewav import v_writewav
        from pyvoicebox.v_readwav import v_readwav
        data = np.zeros((10, 1))
        for fs_in in [8000, 16000, 22050, 44100, 48000]:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmpfile = f.name
            try:
                v_writewav(data, fs_in, tmpfile, 'sp16')
                _, fs_out = v_readwav(tmpfile, 'p')
                assert fs_out == fs_in
            finally:
                os.unlink(tmpfile)


# ============================================================
# v_readhtk / v_writehtk (HTK I/O)
# ============================================================
class TestHtk:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_htk.mat')

    def test_read_user(self):
        """Read USER type HTK file."""
        from pyvoicebox.v_readhtk import v_readhtk
        htk_file = os.path.join(REF_DIR, 'test_user.htk')
        d, fp, dt, tc, t = v_readhtk(htk_file)
        expected = self.ref['htk_user_read']
        np.testing.assert_allclose(d, expected, rtol=1e-5)
        assert abs(fp - float(self.ref['htk_user_fp_read'])) < 1e-10
        assert dt == int(self.ref['htk_user_dt_read'])
        assert tc == int(self.ref['htk_user_tc_read'])

    def test_read_mfcc(self):
        """Read MFCC+E type HTK file."""
        from pyvoicebox.v_readhtk import v_readhtk
        htk_file = os.path.join(REF_DIR, 'test_mfcc.htk')
        d, fp, dt, tc, t = v_readhtk(htk_file)
        expected = self.ref['htk_mfcc_read']
        np.testing.assert_allclose(d, expected, rtol=1e-5)
        assert dt == int(self.ref['htk_mfcc_dt_read'])
        assert tc == int(self.ref['htk_mfcc_tc_read'])

    def test_read_waveform(self):
        """Read WAVEFORM type HTK file."""
        from pyvoicebox.v_readhtk import v_readhtk
        htk_file = os.path.join(REF_DIR, 'test_wave.htk')
        d, fp, dt, tc, t = v_readhtk(htk_file)
        expected = self.ref['htk_wave_read']
        if expected.ndim == 1:
            expected = expected.reshape(-1, 1)
        np.testing.assert_allclose(d, expected, rtol=1e-10)
        assert dt == int(self.ref['htk_wave_dt_read'])
        assert tc == int(self.ref['htk_wave_tc_read'])

    def test_read_plp(self):
        """Read PLP type with modifiers."""
        from pyvoicebox.v_readhtk import v_readhtk
        htk_file = os.path.join(REF_DIR, 'test_plp.htk')
        d, fp, dt, tc, t = v_readhtk(htk_file)
        expected = self.ref['htk_plp_read']
        np.testing.assert_allclose(d, expected, rtol=1e-5)
        assert dt == int(self.ref['htk_plp_dt_read'])
        assert tc == int(self.ref['htk_plp_tc_read'])
        assert t == str(self.ref['htk_plp_t_read'])

    def test_write_read_roundtrip(self):
        """Write HTK file and read it back."""
        from pyvoicebox.v_writehtk import v_writehtk
        from pyvoicebox.v_readhtk import v_readhtk
        rng = np.random.default_rng(42)
        data = rng.standard_normal((25, 10))
        fp = 0.01
        tc = 9  # USER type
        with tempfile.NamedTemporaryFile(suffix='.htk', delete=False) as f:
            tmpfile = f.name
        try:
            v_writehtk(tmpfile, data, fp, tc)
            d, fp_r, dt_r, tc_r, t_r = v_readhtk(tmpfile)
            np.testing.assert_allclose(d, data, rtol=1e-5)
            assert abs(fp_r - fp) < 1e-10
            assert dt_r == 9
            assert tc_r == 9
            assert t_r == 'USER'
        finally:
            os.unlink(tmpfile)

    def test_write_read_waveform_roundtrip(self):
        """Write and read waveform type HTK data."""
        from pyvoicebox.v_writehtk import v_writehtk
        from pyvoicebox.v_readhtk import v_readhtk
        data = np.round(np.random.default_rng(42).standard_normal(200) * 1000)
        fp = 1.0 / 16000
        tc = 0  # WAVEFORM
        with tempfile.NamedTemporaryFile(suffix='.htk', delete=False) as f:
            tmpfile = f.name
        try:
            v_writehtk(tmpfile, data, fp, tc)
            d, fp_r, dt_r, tc_r, t_r = v_readhtk(tmpfile)
            np.testing.assert_allclose(d.ravel(), data.ravel(), rtol=1e-10)
            assert dt_r == 0
            assert t_r == 'WAVEFORM'
        finally:
            os.unlink(tmpfile)
