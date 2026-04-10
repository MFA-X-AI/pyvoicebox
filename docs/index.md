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

## How does it compare to librosa and openSMILE?

pyvoicebox, librosa, and openSMILE cover overlapping but fundamentally different parts of audio processing:

- **pyvoicebox** — speech engineering: LPC, enhancement, quality metrics, classical speech analysis.
- **librosa** — music information retrieval: beat tracking, chroma, CQT, harmonic/percussive separation.
- **openSMILE** — reproducible paralinguistic features for affective computing, with a C++ real-time core.

| | pyvoicebox | librosa | openSMILE |
|---|---|---|---|
| License | LGPL-3.0 | ISC | Dual — free for research, **commercial licence required** from audEERING |
| LPC analysis (61 representations) | Full suite | `lpc()` only | Internal, not exposed |
| Speech enhancement (MMSE, spectral subtraction, dereverb) | Full | None | None |
| Psychoacoustic quality metrics (PESQ, SII, STOI, phon/sone) | Full | None | None |
| Gaussian mixtures (fit, score, merge, divergence) | Full | None | None |
| Pitch detection | PEFAC, RAPT, DYPSA | pYIN | SHS, SWIPE', ACF |
| Standardised feature sets (ComParE, eGeMAPS) | None | None | **Full** |
| MIR features (chroma, CQT, beat tracking) | None | Full | Partial |
| Real-time / embedded deployment | No | No | **Yes** (C++) |
| MFCC / mel spectrogram | Yes | Yes | Yes |

Use **pyvoicebox** when you need speech-specific processing (LPC, enhancement, quality metrics) or are porting MATLAB code that depends on VOICEBOX. Use **librosa** for music information retrieval and quick audio-ML prototyping. Use **openSMILE** when you need reproducible paralinguistic feature sets or real-time deployment — but check the commercial licence if you're not using it for academic research.

## Next Steps

- [Getting Started](getting-started.md) — installation and quick examples
- [API Reference](api/audio-io.md) — browse all 282 functions, organised by topic

## Acknowledgements

pyvoicebox exists only because of the decades of work by **Prof. Mike Brookes** and collaborators at the **Speech and Audio Processing Lab, Imperial College London**. Every function in this package is a direct port of their MATLAB source.

If you use pyvoicebox in academic work, please cite the original toolbox:

> Brookes, M., *VOICEBOX: Speech Processing Toolbox for MATLAB*, Department of Electrical and Electronic Engineering, Imperial College London, 1997–present. <https://github.com/ImperialCollegeLondon/sap-voicebox>

## License

pyvoicebox is distributed under the **GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later)**, matching the upstream MATLAB toolbox.
