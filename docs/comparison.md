---
icon: material/scale-balance
---

# Comparison with librosa and openSMILE

pyvoicebox, librosa, and openSMILE cover overlapping but fundamentally different parts of audio processing:

- **pyvoicebox** — speech engineering: LPC, enhancement, quality metrics, classical speech analysis.
- **librosa** — music information retrieval: beat tracking, chroma, CQT, harmonic/percussive separation.
- **openSMILE** — reproducible paralinguistic features for affective computing, with a C++ real-time core.

## Feature comparison

| | pyvoicebox | librosa | openSMILE |
|---|---|---|---|
| License | LGPL-3.0 | ISC | Dual — free for research, **commercial licence required** from audEERING |
| LPC analysis (60+ representations) | Full suite | `lpc()` only | Internal, not exposed |
| Speech enhancement (MMSE, spectral subtraction, dereverb) | Full | None | None |
| Psychoacoustic quality metrics (PESQ, SII, STOI, phon/sone) | Full | None | None |
| Gaussian mixtures (fit, score, merge, divergence) | Full | None | None |
| Pitch detection | PEFAC, RAPT, DYPSA | pYIN | SHS, SWIPE', ACF |
| Standardised feature sets (ComParE, eGeMAPS) | None | None | **Full** |
| MIR features (chroma, CQT, beat tracking) | None | Full | Partial |
| Real-time / embedded deployment | No | No | **Yes** (C++) |
| MFCC / mel spectrogram | Yes | Yes | Yes |

## When to use which

Use **pyvoicebox** when you need speech-specific processing (LPC, enhancement, quality metrics) or are porting MATLAB code that depends on VOICEBOX.

Use **librosa** for music information retrieval and quick audio-ML prototyping.

Use **openSMILE** when you need reproducible paralinguistic feature sets (ComParE, eGeMAPS) or real-time deployment — but check the commercial licence if you're not using it for academic research.

## Using them together

These tools complement each other. A common pipeline might be:

1. **pyvoicebox** — clean noisy speech with `v_ssubmmse`, estimate noise with `v_estnoiseg`
2. **openSMILE** — extract eGeMAPS features from the cleaned speech
3. **librosa** — generate mel spectrogram features for a CNN classifier
4. **scikit-learn / PyTorch** — train the final model

Or in a speech quality assessment pipeline:

1. **librosa** — load audio from various formats
2. **pyvoicebox** — compute PESQ scores (`v_pesq2mos`), segmental SNR (`v_snrseg`), active speech level (`v_activlev`)
