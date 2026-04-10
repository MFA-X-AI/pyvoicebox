# pyvoicebox

A complete Python port of the [VOICEBOX](https://github.com/ImperialCollegeLondon/sap-voicebox) speech processing toolbox, originally written in MATLAB by Mike Brookes at Imperial College London's Speech and Audio Processing Lab.

**282 functions**, validated against the original MATLAB source via GNU Octave with **511 automated tests**.

## What is VOICEBOX?

VOICEBOX is a comprehensive MATLAB toolkit for speech and audio signal processing maintained since the 1990s. It covers areas that most Python audio libraries don't touch:

- **Linear Predictive Coding** — 61 functions for LPC analysis and conversion between representations (AR coefficients, cepstra, reflection coefficients, line spectra, etc.)
- **Gaussian Mixture Models** — full GMM suite: fitting (EM), scoring, merging, divergence, batch processing
- **Speech Enhancement** — spectral subtraction, MMSE estimators, noise estimation (Rangachari & Loizou, Martin)
- **Pitch Detection** — PEFAC (`v_fxpefac`), RAPT (`v_fxrapt`), DYPSA glottal closure detection
- **Psychoacoustics** — PESQ/MOS mapping, Speech Intelligibility Index (SII), STOI, loudness (phon/sone)
- **Rotations & Quaternions** — Euler angles, rotation matrices, quaternions, polygon/polyhedron geometry
- **Audio Codecs** — mu-law, A-law, WAV, HTK, SPHERE/TIMIT, AIFF, AU, FLAC readers
- **Frequency Scales** — Mel, Bark, ERB, Cent, MIDI conversions
- **Signal Processing** — enframing, overlap-add, STFT, filterbanks, zero-crossing detection, Teager energy

## Why not librosa?

librosa is excellent for music information retrieval (MIR) — beat tracking, chroma features, harmonic/percussive separation. pyvoicebox targets a different domain:

| Area | pyvoicebox | librosa |
|------|-----------|---------|
| LPC analysis (61 representations) | Full suite | `lpc()` only |
| Gaussian mixtures | Complete (fit, score, merge, transform) | None |
| Speech enhancement | MMSE, spectral subtraction, noise estimation | None |
| Pitch detection | PEFAC, RAPT, DYPSA | `pyin` |
| Psychoacoustic metrics | PESQ, SII, STOI, phon/sone | None |
| Quaternion/rotation math | Full | None |
| Audio codecs (HTK, SPHERE, A-law, mu-law) | Full | None |
| Mel spectrogram / MFCC | Yes | Yes |
| Beat tracking / tempo | None | Yes |
| Harmonic/percussive separation | None | Yes |

Use pyvoicebox when you need speech-specific processing, LPC analysis, or are porting MATLAB code that depends on VOICEBOX. Use librosa for MIR tasks. They complement each other.

## Next Steps

- [Getting Started](getting-started.md) — installation and quick examples
- [API Reference](api/audio-io.md) — browse all 282 functions, organised by topic
