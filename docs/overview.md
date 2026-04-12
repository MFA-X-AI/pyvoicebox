---
icon: material/book-open-variant
---

# pyvoicebox

A complete Python port of the [VOICEBOX](https://github.com/ImperialCollegeLondon/sap-voicebox) speech processing toolbox, originally written in MATLAB by Mike Brookes at Imperial College London.

**280+ functions**, fully typed, validated against the original MATLAB source via GNU Octave with **[500+ automated tests](testing.md)**.

## What is VOICEBOX?

VOICEBOX is a comprehensive MATLAB toolkit for speech and audio signal processing maintained since the 1990s. It covers areas that most Python audio libraries don't touch:

- **[LPC Analysis](api/lpc.md)** — 60+ functions for conversion between AR coefficients, cepstra, reflection coefficients, line spectra, and more
- **[Speech Enhancement](api/speech-enhancement.md)** — spectral subtraction, MMSE estimators, noise estimation
- **[Psychoacoustic Metrics](api/speech-analysis.md)** — PESQ/MOS, SII, STOI, phon/sone loudness
- **[Pitch Detection](api/speech-analysis.md)** — PEFAC, RAPT, DYPSA glottal closure detection
- **[Gaussian Mixtures](api/probability.md)** — full GMM suite: fitting (EM), scoring, merging, divergence
- **[Rotations & Quaternions](api/rotation-geometry.md)** — Euler angles, rotation matrices, quaternions, geometry
- **[Audio Codecs](api/audio-io.md)** — WAV, HTK, SPHERE/TIMIT, AIFF, AU, FLAC, A-law, mu-law
- **[Frequency Scales](api/frequency-scales.md)** — Mel, Bark, ERB, Cent, MIDI conversions
- **[Signal Processing](api/signal-processing.md)** — enframing, overlap-add, STFT, filterbanks, Teager energy

See how it [compares to librosa and openSMILE](comparison.md).

## Quick install

```bash
pip install pyvoicebox                # core (numpy, scipy, soundfile)
pip install "pyvoicebox[plot]"        # with matplotlib for plotting functions
```

See the [Getting Started](getting-started.md) guide for examples and usage details.

## Notebooks

Interactive Jupyter notebooks are available in the [`notebooks/`](https://github.com/MFA-X-AI/pyvoicebox/tree/master/notebooks) directory:

- **Frequency Scales** — convert between Hz, Mel, Bark, ERB, Cent, MIDI with visualisations
- **MFCC & Spectrograms** — extract MFCCs, build mel filterbanks, compute spectrograms
- **LPC Analysis** — AR coefficients, spectral envelopes, representation conversions, bandwidth expansion
- **Speech Enhancement** — add noise, measure SNR, compare clean vs noisy spectrograms
