# Notebooks

Interactive Jupyter notebooks demonstrating pyvoicebox with real speech data. Audio data is downloaded automatically on first run.

## Setup

```bash
pip install "py-voicebox[notebooks]"
jupyter notebook notebooks/
```

Or open any notebook directly in Google Colab — each has a `%pip install` cell at the top.

## Notebooks

| Notebook | Description | Colab |
|---|---|---|
| `01_speech_analysis` | Waveform, spectrogram, MFCCs, and pitch tracking | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/01_speech_analysis.ipynb) |
| `02_speech_enhancement` | Add noise at different SNRs, run MMSE enhancement, measure improvement | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/02_speech_enhancement.ipynb) |
| `03_lpc_analysis` | LPC spectral envelopes, order comparison, coefficient conversions, bandwidth expansion | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/03_lpc_analysis.ipynb) |
| `04_speaker_identification` | MFCC extraction, GMM training, speaker classification | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/04_speaker_identification.ipynb) |
| `05_emotion_recognition` | TEO vs MFCC features on EmoDB with Random Forest classification | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/05_emotion_recognition.ipynb) |
