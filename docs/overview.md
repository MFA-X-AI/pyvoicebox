---
icon: material/book-open-variant
---

# pyvoicebox

A complete Python port of the [VOICEBOX](https://github.com/ImperialCollegeLondon/sap-voicebox) Speech and Audio Processing toolbox, originally written in MATLAB by Mike Brookes at Imperial College London.

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
pip install pyvoicebox-sap                # core (numpy, scipy, soundfile)
pip install "pyvoicebox-sap[plot]"        # with matplotlib for plotting functions
```

See the [Getting Started](getting-started.md) guide for examples and usage details.

## Notebooks

Interactive Jupyter notebooks are available in the [`notebooks/`](https://github.com/MFA-X-AI/pyvoicebox/tree/master/notebooks) directory:

| Notebook | Description | Colab |
|---|---|---|
| [Visualize Speech](https://github.com/MFA-X-AI/pyvoicebox/blob/master/notebooks/01_speech_analysis.ipynb) | Waveform, spectrogram, MFCCs, and pitch tracking | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/01_speech_analysis.ipynb) |
| [Clean Up Noisy Speech](https://github.com/MFA-X-AI/pyvoicebox/blob/master/notebooks/02_speech_enhancement.ipynb) | Add noise, run MMSE enhancement, measure SNR improvement | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/02_speech_enhancement.ipynb) |
| [Inside the Vocal Tract](https://github.com/MFA-X-AI/pyvoicebox/blob/master/notebooks/03_lpc_analysis.ipynb) | LPC spectral envelopes, coefficient conversions, bandwidth expansion | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/03_lpc_analysis.ipynb) |
| [Who Said That?](https://github.com/MFA-X-AI/pyvoicebox/blob/master/notebooks/04_speaker_identification.ipynb) | Speaker identification with GMMs | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/04_speaker_identification.ipynb) |
| [Emotion Recognition](https://github.com/MFA-X-AI/pyvoicebox/blob/master/notebooks/05_emotion_recognition.ipynb) | TEO vs MFCC features on EmoDB with Random Forest | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/05_emotion_recognition.ipynb) |
