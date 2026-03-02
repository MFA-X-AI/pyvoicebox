"""Shared pytest fixtures for pyvoicebox test suite.

Clones the original sap-voicebox MATLAB source and generates reference
data on-the-fly via GNU Octave so the repo ships no pre-built .mat files.

Cache location: ~/.cache/pyvoicebox-test/
"""

import os
import shutil
import subprocess
from pathlib import Path

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
    ``octave`` are not installed.
    """
    # --- clone -----------------------------------------------------------
    if not (VOICEBOX_SRC / ".git").exists():
        if not shutil.which("git"):
            pytest.skip("git is not installed -- cannot clone voicebox source")
        VOICEBOX_SRC.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/ImperialCollegeLondon/sap-voicebox.git",
                str(VOICEBOX_SRC),
            ],
            check=True,
        )

    voicebox_m_dir = VOICEBOX_SRC / "voicebox"
    assert voicebox_m_dir.is_dir(), f"voicebox/ not found in {VOICEBOX_SRC}"

    # --- generate ref data -----------------------------------------------
    if not shutil.which("octave"):
        pytest.skip("GNU Octave is not installed -- cannot generate reference data")

    REF_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Skip regeneration when mat files already present
    existing = list(REF_DATA_DIR.glob("*.mat"))
    if len(existing) >= 10:
        return

    for m_file in sorted(HARNESS_DIR.glob("gen_ref_phase*.m")):
        cmd = [
            "octave", "--no-gui", "--silent",
            str(m_file),
            str(voicebox_m_dir),  # arg1: voicebox MATLAB dir
            str(REF_DATA_DIR),    # arg2: output dir
        ]
        # Phase 7 needs normcdf/normpdf stubs
        if STUBS_DIR.is_dir():
            cmd.append(str(STUBS_DIR))  # arg3: stubs dir (optional)
        subprocess.run(cmd, check=True)


@pytest.fixture(scope="session")
def load_ref():
    """Return a callable ``load_ref(name)`` that loads a .mat reference file."""
    ref_dir = str(REF_DATA_DIR)

    def _load(name):
        return loadmat(os.path.join(ref_dir, name), squeeze_me=True)

    return _load
