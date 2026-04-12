"""Shared pytest fixtures for pyvoicebox test suite.

Clones the original sap-voicebox MATLAB source and generates reference
data on-the-fly via GNU Octave so the repo ships no pre-built .mat files.

Cache location: ~/.cache/pyvoicebox-test/
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

CACHE_DIR = Path.home() / ".cache" / "pyvoicebox-test"
VOICEBOX_SRC = CACHE_DIR / "voicebox-src"
REF_DATA_DIR = CACHE_DIR / "ref_data"

HARNESS_DIR = Path(__file__).parent / "octave_harness"
STUBS_DIR = Path(__file__).parent / "tmp"  # normcdf/normpdf stubs for Octave


@pytest.fixture(scope="session", autouse=True)
def _generate_ref_data():
    """Clone voicebox MATLAB source and generate reference .mat files.

    Everything is cached under ``~/.cache/pyvoicebox-test/``.
    Skips the entire test session with a clear message when ``git`` or
    ``octave`` are not installed, or when the cache cannot be created.
    """
    # --- create cache dir --------------------------------------------------
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        pytest.skip(f"Cannot create cache directory {CACHE_DIR}: {e}")

    # --- clone -------------------------------------------------------------
    if not (VOICEBOX_SRC / ".git").exists():
        if not shutil.which("git"):
            pytest.skip("git is not installed -- cannot clone voicebox source")
        try:
            VOICEBOX_SRC.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    "https://github.com/ImperialCollegeLondon/sap-voicebox.git",
                    str(VOICEBOX_SRC),
                ],
                check=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("git clone timed out after 120s")
        except subprocess.CalledProcessError as e:
            pytest.skip(f"git clone failed: {e}")
        except (PermissionError, OSError) as e:
            pytest.skip(f"Cannot create voicebox source directory: {e}")

    voicebox_m_dir = VOICEBOX_SRC / "voicebox"
    assert voicebox_m_dir.is_dir(), f"voicebox/ not found in {VOICEBOX_SRC}"

    # --- generate ref data -------------------------------------------------
    if not shutil.which("octave"):
        pytest.skip("GNU Octave is not installed -- cannot generate reference data")

    try:
        REF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        pytest.skip(f"Cannot create ref data directory {REF_DATA_DIR}: {e}")

    # Skip regeneration when enough .mat files already exist
    # (one per harness script at minimum)
    harness_scripts = sorted(HARNESS_DIR.glob("gen_ref_phase*.m"))
    existing = list(REF_DATA_DIR.glob("*.mat"))
    if len(existing) >= len(harness_scripts):
        return

    for m_file in harness_scripts:
        cmd = [
            "octave", "--no-gui", "--silent",
            str(m_file),
            str(voicebox_m_dir),  # arg1: voicebox MATLAB dir
            str(REF_DATA_DIR),    # arg2: output dir
        ]
        # Phase 7 needs normcdf/normpdf stubs
        if STUBS_DIR.is_dir():
            cmd.append(str(STUBS_DIR))  # arg3: stubs dir (optional)
        try:
            subprocess.run(cmd, check=True, timeout=300)
        except subprocess.TimeoutExpired:
            pytest.skip(f"Octave harness {m_file.name} timed out after 300s")
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Octave harness {m_file.name} failed: {e}")


@pytest.fixture(scope="session")
def load_ref():
    """Return a callable ``load_ref(name)`` that loads a .mat reference file."""
    ref_dir = str(REF_DATA_DIR)

    def _load(name):
        return loadmat(os.path.join(ref_dir, name), squeeze_me=True)

    return _load


# ── Synthetic signal helpers (used by smoke tests) ──

def _sine(fs=16000, dur=0.5, f0=440):
    """Generate a simple sine wave."""
    t = np.arange(int(fs * dur)) / fs
    return np.sin(2 * np.pi * f0 * t), fs


def _speech_like(fs=16000, dur=0.5):
    """Generate a speech-like signal with harmonics."""
    t = np.arange(int(fs * dur)) / fs
    sig = np.zeros_like(t)
    for k in range(1, 20):
        freq = k * 150
        if freq > fs / 2:
            break
        sig += (1.0 / k) * np.sin(2 * np.pi * freq * t)
    sig = sig / np.max(np.abs(sig)) * 0.8
    sig += 0.01 * np.random.randn(len(sig))
    return sig, fs


def _lpc_ar(order=12):
    """Get LPC AR coefficients from a test signal."""
    from pyvoicebox.v_lpcauto import v_lpcauto
    sig, fs = _sine()
    ar, e, k = v_lpcauto(sig, order)
    return ar
