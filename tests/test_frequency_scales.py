"""Tests for frequency scale conversion functions."""

import os
import numpy as np
import pytest
from scipy.io import loadmat

REF_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pyvoicebox-test', 'ref_data')


def load_ref(name):
    return loadmat(os.path.join(REF_DIR, name), squeeze_me=True)


# ============================================================
# v_frq2mel
# ============================================================
class TestFrq2Mel:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_frq2mel.mat')

    def test_mel_values(self):
        from pyvoicebox.v_frq2mel import v_frq2mel
        frq = self.ref['frq_mel']
        mel, mr = v_frq2mel(frq)
        np.testing.assert_allclose(mel, self.ref['mel_out'], rtol=1e-10)

    def test_mel_gradient(self):
        from pyvoicebox.v_frq2mel import v_frq2mel
        frq = self.ref['frq_mel']
        mel, mr = v_frq2mel(frq)
        np.testing.assert_allclose(mr, self.ref['mel_mr'], rtol=1e-10)

    def test_mel_1000(self):
        """mel(1000 Hz) should be 1000."""
        from pyvoicebox.v_frq2mel import v_frq2mel
        mel, _ = v_frq2mel(np.array([1000.0]))
        np.testing.assert_allclose(mel, 1000.0, rtol=1e-10)

    def test_mel_zero(self):
        """mel(0 Hz) should be 0."""
        from pyvoicebox.v_frq2mel import v_frq2mel
        mel, _ = v_frq2mel(np.array([0.0]))
        assert mel[0] == 0.0


# ============================================================
# v_mel2frq
# ============================================================
class TestMel2Frq:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_mel2frq.mat')

    def test_frq_values(self):
        from pyvoicebox.v_mel2frq import v_mel2frq
        mel = self.ref['mel_in']
        frq, mr = v_mel2frq(mel)
        np.testing.assert_allclose(frq, self.ref['mel2frq_out'], rtol=1e-10)

    def test_frq_gradient(self):
        from pyvoicebox.v_mel2frq import v_mel2frq
        mel = self.ref['mel_in']
        frq, mr = v_mel2frq(mel)
        np.testing.assert_allclose(mr, self.ref['mel2frq_mr'], rtol=1e-10)

    def test_roundtrip(self):
        """frq -> mel -> frq should round-trip."""
        from pyvoicebox.v_frq2mel import v_frq2mel
        from pyvoicebox.v_mel2frq import v_mel2frq
        frq_orig = np.array([100.0, 500.0, 1000.0, 4000.0])
        mel, _ = v_frq2mel(frq_orig)
        frq_back, _ = v_mel2frq(mel)
        np.testing.assert_allclose(frq_back, frq_orig, rtol=1e-10)


# ============================================================
# v_frq2bark
# ============================================================
class TestFrq2Bark:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_frq2bark.mat')

    def test_default_bark(self):
        from pyvoicebox.v_frq2bark import v_frq2bark
        frq = self.ref['frq_bark']
        b, c = v_frq2bark(frq)
        np.testing.assert_allclose(b, self.ref['bark_def'], rtol=1e-10)

    def test_default_bandwidth(self):
        from pyvoicebox.v_frq2bark import v_frq2bark
        frq = self.ref['frq_bark']
        b, c = v_frq2bark(frq)
        np.testing.assert_allclose(c, self.ref['bark_def_c'], rtol=1e-10)

    def test_zwicker(self):
        from pyvoicebox.v_frq2bark import v_frq2bark
        frq = self.ref['frq_bark']
        b, c = v_frq2bark(frq, 'z')
        np.testing.assert_allclose(b, self.ref['bark_z'], rtol=1e-10)
        np.testing.assert_allclose(c, self.ref['bark_z_c'], rtol=1e-10)

    def test_schroeder(self):
        from pyvoicebox.v_frq2bark import v_frq2bark
        frq = self.ref['frq_bark']
        b, c = v_frq2bark(frq, 's')
        np.testing.assert_allclose(b, self.ref['bark_s'], rtol=1e-10)
        np.testing.assert_allclose(c, self.ref['bark_s_c'], rtol=1e-10)

    def test_traunmuller_lh(self):
        from pyvoicebox.v_frq2bark import v_frq2bark
        frq = self.ref['frq_bark']
        b, c = v_frq2bark(frq, 'lh')
        np.testing.assert_allclose(b, self.ref['bark_lh'], rtol=1e-10)
        np.testing.assert_allclose(c, self.ref['bark_lh_c'], rtol=1e-10)

    def test_no_corrections(self):
        from pyvoicebox.v_frq2bark import v_frq2bark
        frq = self.ref['frq_bark']
        b, c = v_frq2bark(frq, 'LH')
        np.testing.assert_allclose(b, self.ref['bark_LH'], rtol=1e-10)
        np.testing.assert_allclose(c, self.ref['bark_LH_c'], rtol=1e-10)

    def test_unipolar(self):
        from pyvoicebox.v_frq2bark import v_frq2bark
        frq = self.ref['frq_bark_u']
        b, c = v_frq2bark(frq, 'u')
        np.testing.assert_allclose(b, self.ref['bark_u'], rtol=1e-10)
        np.testing.assert_allclose(c, self.ref['bark_u_c'], rtol=1e-10)


# ============================================================
# v_bark2frq
# ============================================================
class TestBark2Frq:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_bark2frq.mat')

    def test_default_frq(self):
        from pyvoicebox.v_bark2frq import v_bark2frq
        b = self.ref['bark_in']
        f, c = v_bark2frq(b)
        np.testing.assert_allclose(f, self.ref['b2f_def'], rtol=1e-10)

    def test_default_bandwidth(self):
        from pyvoicebox.v_bark2frq import v_bark2frq
        b = self.ref['bark_in']
        f, c = v_bark2frq(b)
        np.testing.assert_allclose(c, self.ref['b2f_def_c'], rtol=1e-10)

    def test_schroeder(self):
        from pyvoicebox.v_bark2frq import v_bark2frq
        b = self.ref['bark_in']
        f, c = v_bark2frq(b, 's')
        np.testing.assert_allclose(f, self.ref['b2f_s'], rtol=1e-10)
        np.testing.assert_allclose(c, self.ref['b2f_s_c'], rtol=1e-10)

    def test_traunmuller_lh(self):
        from pyvoicebox.v_bark2frq import v_bark2frq
        b = self.ref['bark_in']
        f, c = v_bark2frq(b, 'lh')
        np.testing.assert_allclose(f, self.ref['b2f_lh'], rtol=1e-10)
        np.testing.assert_allclose(c, self.ref['b2f_lh_c'], rtol=1e-10)

    def test_no_corrections(self):
        from pyvoicebox.v_bark2frq import v_bark2frq
        b = self.ref['bark_in']
        f, c = v_bark2frq(b, 'LH')
        np.testing.assert_allclose(f, self.ref['b2f_LH'], rtol=1e-10)
        np.testing.assert_allclose(c, self.ref['b2f_LH_c'], rtol=1e-10)

    def test_unipolar(self):
        from pyvoicebox.v_bark2frq import v_bark2frq
        b = self.ref['bark_in_u']
        f, c = v_bark2frq(b, 'u')
        np.testing.assert_allclose(f, self.ref['b2f_u'], rtol=1e-10)
        np.testing.assert_allclose(c, self.ref['b2f_u_c'], rtol=1e-10)

    def test_roundtrip(self):
        """frq -> bark -> frq should round-trip."""
        from pyvoicebox.v_frq2bark import v_frq2bark
        from pyvoicebox.v_bark2frq import v_bark2frq
        frq_orig = np.array([100.0, 500.0, 1000.0, 4000.0])
        b, _ = v_frq2bark(frq_orig)
        frq_back, _ = v_bark2frq(b)
        np.testing.assert_allclose(frq_back, frq_orig, rtol=1e-8)


# ============================================================
# v_frq2erb
# ============================================================
class TestFrq2Erb:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_frq2erb.mat')

    def test_erb_values(self):
        from pyvoicebox.v_frq2erb import v_frq2erb
        frq = self.ref['frq_erb']
        erb, bnd = v_frq2erb(frq)
        np.testing.assert_allclose(erb, self.ref['erb_out'], rtol=1e-10)

    def test_erb_bandwidth(self):
        from pyvoicebox.v_frq2erb import v_frq2erb
        frq = self.ref['frq_erb']
        erb, bnd = v_frq2erb(frq)
        np.testing.assert_allclose(bnd, self.ref['erb_bnd'], rtol=1e-10)


# ============================================================
# v_erb2frq
# ============================================================
class TestErb2Frq:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_erb2frq.mat')

    def test_frq_values(self):
        from pyvoicebox.v_erb2frq import v_erb2frq
        erb = self.ref['erb_in']
        frq, bnd = v_erb2frq(erb)
        np.testing.assert_allclose(frq, self.ref['e2f_out'], rtol=1e-10)

    def test_frq_bandwidth(self):
        from pyvoicebox.v_erb2frq import v_erb2frq
        erb = self.ref['erb_in']
        frq, bnd = v_erb2frq(erb)
        np.testing.assert_allclose(bnd, self.ref['e2f_bnd'], rtol=1e-10)

    def test_roundtrip(self):
        """frq -> erb -> frq should round-trip."""
        from pyvoicebox.v_frq2erb import v_frq2erb
        from pyvoicebox.v_erb2frq import v_erb2frq
        frq_orig = np.array([100.0, 500.0, 1000.0, 4000.0])
        erb, _ = v_frq2erb(frq_orig)
        frq_back, _ = v_erb2frq(erb)
        np.testing.assert_allclose(frq_back, frq_orig, rtol=1e-8)


# ============================================================
# v_frq2cent
# ============================================================
class TestFrq2Cent:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_frq2cent.mat')

    def test_cent_values(self):
        from pyvoicebox.v_frq2cent import v_frq2cent
        frq = self.ref['frq_cent']
        c, cr = v_frq2cent(frq)
        np.testing.assert_allclose(c, self.ref['cent_out'], rtol=1e-10)

    def test_cent_gradient(self):
        from pyvoicebox.v_frq2cent import v_frq2cent
        frq = self.ref['frq_cent']
        c, cr = v_frq2cent(frq)
        np.testing.assert_allclose(cr, self.ref['cent_cr'], rtol=1e-10)

    def test_440_is_5700(self):
        """440 Hz should correspond to 5700 cents."""
        from pyvoicebox.v_frq2cent import v_frq2cent
        c, _ = v_frq2cent(np.array([440.0]))
        np.testing.assert_allclose(c, 5700.0, rtol=1e-10)


# ============================================================
# v_cent2frq
# ============================================================
class TestCent2Frq:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_cent2frq.mat')

    def test_frq_values(self):
        from pyvoicebox.v_cent2frq import v_cent2frq
        c = self.ref['cent_in']
        frq, cr = v_cent2frq(c)
        np.testing.assert_allclose(frq, self.ref['c2f_out'], rtol=1e-10)

    def test_frq_gradient(self):
        from pyvoicebox.v_cent2frq import v_cent2frq
        c = self.ref['cent_in']
        frq, cr = v_cent2frq(c)
        np.testing.assert_allclose(cr, self.ref['c2f_cr'], rtol=1e-10)

    def test_roundtrip(self):
        """frq -> cent -> frq should round-trip."""
        from pyvoicebox.v_frq2cent import v_frq2cent
        from pyvoicebox.v_cent2frq import v_cent2frq
        frq_orig = np.array([100.0, 440.0, 1000.0, 4000.0])
        c, _ = v_frq2cent(frq_orig)
        frq_back, _ = v_cent2frq(c)
        np.testing.assert_allclose(frq_back, frq_orig, rtol=1e-10)


# ============================================================
# v_frq2midi
# ============================================================
class TestFrq2Midi:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_frq2midi.mat')

    def test_midi_values(self):
        from pyvoicebox.v_frq2midi import v_frq2midi
        frq = self.ref['frq_midi']
        n, t = v_frq2midi(frq)
        np.testing.assert_allclose(n, self.ref['midi_out'], rtol=1e-10)

    def test_midi_text(self):
        from pyvoicebox.v_frq2midi import v_frq2midi
        frq = self.ref['frq_midi']
        n, t = v_frq2midi(frq)
        ref_text = self.ref['midi_text']
        # Reference text is a char array from MATLAB, each row is a note
        for i in range(len(t)):
            assert t[i] == str(ref_text[i]).strip() or t[i].strip() == str(ref_text[i]).strip()

    def test_440_is_69(self):
        """440 Hz should be MIDI note 69."""
        from pyvoicebox.v_frq2midi import v_frq2midi
        n, _ = v_frq2midi(np.array([440.0]))
        np.testing.assert_allclose(n, 69.0, rtol=1e-10)


# ============================================================
# v_midi2frq
# ============================================================
class TestMidi2Frq:
    @classmethod
    def setup_class(cls):
        cls.ref = load_ref('ref_midi2frq.mat')

    def test_equal_tempered(self):
        from pyvoicebox.v_midi2frq import v_midi2frq
        midi = self.ref['midi_in']
        f = v_midi2frq(midi)
        np.testing.assert_allclose(f, self.ref['midi2frq_e'], rtol=1e-10)

    def test_pythagorean(self):
        from pyvoicebox.v_midi2frq import v_midi2frq
        midi = self.ref['midi_in']
        f = v_midi2frq(midi, 'p')
        np.testing.assert_allclose(f, self.ref['midi2frq_p'], rtol=1e-10)

    def test_just_intonation(self):
        from pyvoicebox.v_midi2frq import v_midi2frq
        midi = self.ref['midi_in']
        f = v_midi2frq(midi, 'j')
        np.testing.assert_allclose(f, self.ref['midi2frq_j'], rtol=1e-10)

    def test_pythagorean_fractional(self):
        from pyvoicebox.v_midi2frq import v_midi2frq
        midi = self.ref['midi_frac']
        f = v_midi2frq(midi, 'p')
        np.testing.assert_allclose(f, self.ref['midi2frq_p_frac'], rtol=1e-10)

    def test_just_intonation_fractional(self):
        from pyvoicebox.v_midi2frq import v_midi2frq
        midi = self.ref['midi_frac']
        f = v_midi2frq(midi, 'j')
        np.testing.assert_allclose(f, self.ref['midi2frq_j_frac'], rtol=1e-10)

    def test_69_is_440(self):
        """MIDI note 69 should be 440 Hz."""
        from pyvoicebox.v_midi2frq import v_midi2frq
        f = v_midi2frq(np.array([69.0]))
        np.testing.assert_allclose(f, 440.0, rtol=1e-10)

    def test_roundtrip(self):
        """frq -> midi -> frq should round-trip for equal tempered."""
        from pyvoicebox.v_frq2midi import v_frq2midi
        from pyvoicebox.v_midi2frq import v_midi2frq
        frq_orig = np.array([261.63, 440.0, 880.0])
        n, _ = v_frq2midi(frq_orig)
        frq_back = v_midi2frq(n)
        np.testing.assert_allclose(frq_back, frq_orig, rtol=1e-10)
