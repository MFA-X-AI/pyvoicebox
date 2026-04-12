# Examples

Interactive Jupyter notebooks demonstrating pyvoicebox with real speech data. Audio is downloaded automatically on first run.

Each notebook has a `%pip install` cell at the top — uncomment it to run on Google Colab.

| Notebook | Description | Colab |
|---|---|---|
| [Visualize Speech](speech-analysis.md) | Waveform, spectrogram, MFCCs, and pitch tracking | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/01_speech_analysis.ipynb) |
| [Clean Up Noisy Speech](speech-enhancement.md) | Add noise, run MMSE enhancement, measure SNR | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/02_speech_enhancement.ipynb) |
| [Inside the Vocal Tract](lpc-analysis.md) | LPC spectral envelopes, coefficient conversions | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/03_lpc_analysis.ipynb) |
| [Speaker Identification](speaker-identification.md) | GMM-based speaker classification | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/04_speaker_identification.ipynb) |
| [Emotion Recognition](emotion-recognition.md) | TEO vs MFCC on EmoDB with Random Forest | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/05_emotion_recognition.ipynb) |

## Running locally

```bash
pip install "pyvoicebox[notebooks]"
jupyter notebook notebooks/
```
