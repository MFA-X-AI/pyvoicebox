"""Shared conversion from libsndfile's normalized samples to raw values."""

import soundfile as sf


# libsndfile normalizes integer samples by their signed full-scale range.
# VOICEBOX represents decoded A-law/mu-law samples in 13/14-bit units.
_RAW_SCALE = {
    'PCM_S8': 2**7,
    'PCM_U8': 2**7,
    'PCM_16': 2**15,
    'PCM_24': 2**23,
    'PCM_32': 2**31,
    'ALAW': 2**12,
    'ULAW': 2**13,
}


def _read_raw(filename, subtype, **kwargs):
    """Read samples in their native integer units, preserving floating audio."""
    samples, fs = sf.read(filename, dtype='float64', always_2d=True, **kwargs)
    return samples * _RAW_SCALE.get(subtype, 1), fs
