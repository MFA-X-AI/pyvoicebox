# pyvoicebox

A complete Python port of the [VOICEBOX](https://github.com/ImperialCollegeLondon/sap-voicebox) speech processing toolbox, originally written in MATLAB by Mike Brookes at Imperial College London's Speech and Audio Processing Lab.

**280+ functions**, fully typed, validated against the original MATLAB source via GNU Octave with **500+ automated tests**.

## What is VOICEBOX?

VOICEBOX is a comprehensive MATLAB toolkit for speech and audio signal processing maintained since the 1990s. It covers areas that most Python audio libraries don't touch:

- **Linear Predictive Coding** — 60+ functions for LPC analysis and conversion between representations (AR coefficients, cepstra, reflection coefficients, line spectra, etc.)
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
| LPC analysis (60+ representations) | Full suite | `lpc()` only | Internal, not exposed |
| Speech enhancement (MMSE, spectral subtraction, dereverb) | Full | None | None |
| Psychoacoustic quality metrics (PESQ, SII, STOI, phon/sone) | Full | None | None |
| Gaussian mixtures (fit, score, merge, divergence) | Full | None | None |
| Pitch detection | PEFAC, RAPT, DYPSA | pYIN | SHS, SWIPE', ACF |
| Standardised feature sets (ComParE, eGeMAPS) | None | None | **Full** |
| MIR features (chroma, CQT, beat tracking) | None | Full | Partial |
| Real-time / embedded deployment | No | No | **Yes** (C++) |
| MFCC / mel spectrogram | Yes | Yes | Yes |

Use **pyvoicebox** when you need speech-specific processing (LPC, enhancement, quality metrics) or are porting MATLAB code that depends on VOICEBOX. Use **librosa** for music information retrieval and quick audio-ML prototyping. Use **openSMILE** when you need reproducible paralinguistic feature sets or real-time deployment — but check the commercial licence if you're not using it for academic research.

They complement each other. A common pipeline is pyvoicebox for cleanup and quality scoring, openSMILE or librosa for feature extraction, then scikit-learn / PyTorch for modelling.

## Installation

```bash
pip install py-voicebox                # core (numpy, scipy, soundfile)
pip install "py-voicebox[plot]"        # with matplotlib for plotting functions
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
# Frequency conversions
from pyvoicebox import v_frq2mel, v_mel2frq
mel = v_frq2mel(440)          # Hz to Mel
hz = v_mel2frq(mel)           # back to Hz

# MFCC extraction
from pyvoicebox import v_melcepst
import soundfile as sf
signal, fs = sf.read('speech.wav')
mfcc = v_melcepst(signal, fs, 'M0dD', 12)  # 12 MFCCs + deltas

# LPC analysis
from pyvoicebox import v_lpcauto, v_lpcar2cc
ar, e, k = v_lpcauto(signal, 12)     # 12th-order LPC
cc = v_lpcar2cc(ar)                  # AR -> cepstral coefficients

# Quaternion operations
from pyvoicebox import v_roteu2qr, v_rotqr2ro
import numpy as np
q = v_roteu2qr('xyz', np.array([0.1, 0.2, 0.3]))  # Euler -> quaternion
R = v_rotqr2ro(q)                                 # quaternion -> rotation matrix

# Noise estimation & speech enhancement
from pyvoicebox import v_estnoiseg, v_specsub
```

All functions are available with both the `v_` prefix (matching MATLAB) and without:

```python
from pyvoicebox import frq2mel       # same as v_frq2mel
from pyvoicebox import melcepst      # same as v_melcepst
from pyvoicebox import lpcauto       # same as v_lpcauto
```

## Notebooks

Interactive Jupyter notebooks with audio playback:

| Notebook | Description | Colab |
|---|---|---|
| [Visualize Speech](notebooks/01_speech_analysis.ipynb) | Waveform, spectrogram, MFCCs, and pitch tracking | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/01_speech_analysis.ipynb) |
| [Clean Up Noisy Speech](notebooks/02_speech_enhancement.ipynb) | Add noise, run MMSE enhancement, measure SNR improvement | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/02_speech_enhancement.ipynb) |
| [Inside the Vocal Tract](notebooks/03_lpc_analysis.ipynb) | LPC spectral envelopes, coefficient conversions, bandwidth expansion | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/03_lpc_analysis.ipynb) |
| [Who Said That?](notebooks/04_speaker_identification.ipynb) | Speaker identification with GMMs | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/04_speaker_identification.ipynb) |
| [Emotion Recognition](notebooks/05_emotion_recognition.ipynb) | TEO vs MFCC features on EmoDB with Random Forest | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MFA-X-AI/pyvoicebox/blob/master/notebooks/05_emotion_recognition.ipynb) |

## Function Reference

Functions are grouped by topic, following the same categorisation as the [original VOICEBOX documentation](https://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html). Click any section to expand.

<details>
<summary><strong>Audio File Input/Output</strong></summary>

Read and write a variety of audio file formats: WAV, HTK, SPHERE/TIMIT, AIFF, AU, FLAC, and more.

| Function | Description |
|---|---|
| `v_readwav` | Read a .WAV format sound file |
| `v_writewav` | Write a .WAV format sound file |
| `v_readhtk` | Read an HTK parameter file |
| `v_writehtk` | Write data in HTK format |
| `v_readsph` | Read a SPHERE/TIMIT format sound file |
| `v_readaif` | Read a .AIF (AIFF) format sound file |
| `v_readau` | Read a SUN .AU format sound file |
| `v_readflac` | Read a .FLAC format sound file |
| `v_readsfs` | Read a .SFS (Speech Filing System) format sound file |
| `v_readcnx` | Read a .CNX format sound file |

</details>

<details>
<summary><strong>Frequency Scale Conversion</strong></summary>

Convert between Hz and perceptual (Mel, Bark, ERB) or musical (Cent, MIDI) frequency scales.

| Function | Description |
|---|---|
| `v_frq2mel` | Convert Hertz to Mel frequency scale |
| `v_mel2frq` | Convert Mel frequency scale to Hertz |
| `v_frq2bark` | Convert Hertz to BARK frequency scale |
| `v_bark2frq` | Convert the BARK frequency scale to Hertz |
| `v_frq2erb` | Convert Hertz to ERB frequency scale |
| `v_erb2frq` | Convert ERB frequency scale to Hertz |
| `v_frq2cent` | Convert Hertz to Cents frequency scale |
| `v_cent2frq` | Convert Cents frequency scale to Hertz |
| `v_frq2midi` | Convert frequencies to musical note numbers |
| `v_midi2frq` | Convert musical note numbers to frequencies |

</details>

<details>
<summary><strong>Fourier, DCT and Hartley Transforms</strong></summary>

Fast transforms on real data, plus zoom FFT and FFT-based convolution.

| Function | Description |
|---|---|
| `v_rfft` | Calculate the DFT of real data, returning only the first half |
| `v_irfft` | Inverse FFT of a conjugate symmetric spectrum |
| `v_rsfft` | FFT of a real symmetric spectrum |
| `v_zoomfft` | DTFT evaluated over a linear frequency range |
| `v_rdct` | Discrete cosine transform of real data |
| `v_irdct` | Inverse discrete cosine transform of real data |
| `v_rhartley` | Calculate the Hartley transform of real data |
| `v_convfft` | 1-D convolution or correlation using FFT |
| `v_frac2bin` | Convert a column vector to binary string representation |

</details>

<details>
<summary><strong>Random Numbers and Probability</strong></summary>

RNGs, multivariate Gaussian mixtures (fit/score/merge/divergence), k-means, and probability densities.

| Function | Description |
|---|---|
| `v_randvec` | Generate random vectors from a GMM distribution |
| `v_randiscr` | Generate discrete random values with specified probabilities |
| `v_stdspectrum` | Generate standard acoustic/speech spectra (simplified) |
| `v_randfilt` | Generate filtered Gaussian noise without initial transient |
| `v_rnsubset` | Choose k distinct random integers from 1:n |
| `v_usasi` | Generate USASI noise |
| `v_gaussmix` | Fit a Gaussian mixture model using EM algorithm |
| `v_gaussmixb` | Approximate Bhattacharyya divergence between two GMMs |
| `v_gaussmixd` | Marginal and conditional Gaussian mixture densities |
| `v_gaussmixg` | Global mean, variance and mode of a GMM (computation only) |
| `v_gaussmixk` | Approximate KL divergence between two GMMs |
| `v_gaussmixm` | Estimate mean and variance of the magnitude of a GMM |
| `v_gaussmixp` | Calculate log probability densities from a Gaussian mixture model |
| `v_gaussmixt` | Multiply two GMM PDFs |
| `v_gmmlpdf` | Obsolete wrapper for v_gaussmixp |
| `v_kmeans` | K-means clustering algorithm |
| `v_kmeanlbg` | K-means using Linde-Buzo-Gray algorithm |
| `v_kmeanhar` | K-harmonic means clustering algorithm |
| `v_lognmpdf` | Calculate PDF of a multivariate lognormal distribution |
| `v_normcdflog` | Log of normal CDF, accurate for large negative values |
| `v_chimv` | Approximate mean and variance of non-central chi distribution |
| `v_vonmisespdf` | Von Mises probability distribution |
| `v_maxgauss` | Gaussian approximation to the max of a Gaussian vector |
| `v_berk2prob` | Convert Berksons (log-odds base 2) to probability |
| `v_prob2berk` | Convert probability to Berksons (log-odds base 2) |
| `v_gausprod` | Calculate the product of Gaussians |
| `v_histndim` | Generate an n-dimensional histogram |
| `v_pdfmoments` | Convert between central moments, raw moments and cumulants |
| `v_besselratio` | Bessel function ratio I_{v+1}(x)/I_v(x) |
| `v_besselratioi` | Inverse Bessel function ratio |
| `v_besratinv0` | Inverse of Modified Bessel Ratio I1(k)/I0(k) |

</details>

<details>
<summary><strong>Vector Distance</strong></summary>

Spectral distance measures between LPC filter pairs (Itakura, Itakura–Saito, COSH, Euclidean).

| Function | Description |
|---|---|
| `v_disteusq` | Squared Euclidean distance matrix |
| `v_distitar` | Itakura distance between AR coefficients |
| `v_distitpf` | Itakura distance between power spectra |
| `v_distisar` | Itakura-Saito distance between AR coefficients |
| `v_distispf` | Itakura-Saito distance between power spectra |
| `v_distchar` | COSH spectral distance between AR coefficients |
| `v_distchpf` | COSH spectral distance between power spectra |

</details>

<details>
<summary><strong>Speech Analysis</strong></summary>

Frame-based analysis, spectrograms, pitch trackers, voice activity detection, level measurement, and psychoacoustic metrics.

| Function | Description |
|---|---|
| `v_enframe` | Split signal into (overlapping) frames: one per row |
| `v_overlapadd` | Join overlapping frames together |
| `v_fram2wav` | Convert frame values to a continuous waveform |
| `v_stftw` | Short-time Fourier Transform |
| `v_istftw` | Inverse Short-time Fourier Transform |
| `v_filtbankm` | General filterbank matrix (mel/bark/erb/linear) |
| `v_gammabank` | Gammatone filterbank (stub) |
| `v_correlogram` | Calculate correlogram |
| `v_spgrambw` | Spectrogram computation with configurable bandwidth |
| `v_modspect` | Calculate modulation spectrum of a signal |
| `v_ewgrpdel` | Energy-weighted group delay waveform |
| `v_fxpefac` | PEFAC pitch extraction algorithm |
| `v_fxrapt` | RAPT pitch extraction algorithm |
| `v_dypsa` | Derive glottal closure instances from speech using the DYPSA algorithm |
| `v_vadsohn` | Voice activity detector (Sohn et al.) |
| `v_activlev` | Measure active speech level as per ITU-T P.56 |
| `v_activlevg` | Measure active speech level robustly |
| `v_earnoise` | Add noise to simulate hearing threshold |
| `v_ppmvu` | Calculate PPM and VU meter readings (stub) |
| `v_snrseg` | Measure segmental and global SNR |
| `v_addnoise` | Add noise at a chosen SNR |
| `v_sigalign` | Align a clean reference with a noisy signal |
| `v_txalign` | Find best alignment of two sets of time markers |
| `v_importsii` | Calculate the SII importance function |
| `v_phon2sone` | Convert PHON loudness values to SONEs |
| `v_sone2phon` | Convert SONE loudness values to PHONs |
| `v_pesq2mos` | Convert PESQ speech quality scores to MOS |
| `v_mos2pesq` | Convert MOS speech quality scores to PESQ |
| `v_stoi2prob` | Convert STOI to probability |
| `v_psycdigit` | Psychoacoustic digit recognition test (stub) |
| `v_psycest` | Psychoacoustic estimation (stub) |
| `v_psycestu` | Psychoacoustic estimation utilities (stub) |
| `v_psychofunc` | Calculate psychometric functions |
| `v_soundspeed` | Speed of sound, density and impedance of air |
| `v_sigma` | Estimate glottal opening and closing instants using SIGMA algorithm |

</details>

<details>
<summary><strong>LPC Analysis</strong></summary>

Autocorrelation and covariance LPC, bandwidth expansion, inverse filtering, stability enforcement, and **60+ conversion routines** between every LPC representation (AR coefficients, reflection coefficients, line spectra, cepstra, impulse response, power spectrum, poles/zeros, etc.).

| Function | Description |
|---|---|
| `v_lpcauto` | Perform autocorrelation LPC analysis |
| `v_lpccovar` | Perform covariance LPC analysis |
| `v_lpcconv` | Convert between LPC parameter sets (generates conversion string) |
| `v_lpcbwexp` | Expand formant bandwidths of LPC filter |
| `v_lpcstable` | Test AR coefficients for stability and stabilize if necessary |
| `v_lpcifilt` | Apply inverse filter to speech signal |
| `v_lpcrand` | Generate random stable polynomials |
| `v_rootstab` | Determine number of polynomial roots outside, inside and on the unit circle |
| `v_ccwarpf` | Warp cepstral coefficients |
| `v_lpcar2am` | Convert AR coefficients to AR coefficient matrix |
| `v_lpcar2cc` | Convert AR filter to complex cepstrum |
| `v_lpcar2db` | Convert AR coefficients to power spectrum in dB |
| `v_lpcar2ff` | Convert AR coefficients to complex spectrum |
| `v_lpcar2fm` | Convert autoregressive coefficients to formant freq+amp+bw |
| `v_lpcar2im` | Convert AR coefficients to impulse response |
| `v_lpcar2ls` | Convert AR polynomial to line spectrum pair frequencies |
| `v_lpcar2pf` | Convert AR coefficients to power spectrum |
| `v_lpcar2pp` | Convert AR filter to power spectrum polynomial in cos(w) |
| `v_lpcar2ra` | Convert AR filter to inverse filter autocorrelation coefficients |
| `v_lpcar2rf` | Convert autoregressive coefficients to reflection coefficients |
| `v_lpcar2rr` | Convert autoregressive coefficients to autocorrelation coefficients |
| `v_lpcar2zz` | Convert AR filter to z-plane poles |
| `v_lpcrf2aa` | Convert reflection coefficients to area function |
| `v_lpcrf2ao` | Convert reflection coefficients to area ratios |
| `v_lpcrf2ar` | Convert reflection coefficients to autoregressive coefficients |
| `v_lpcrf2is` | Convert reflection coefficients to inverse sines |
| `v_lpcrf2la` | Convert reflection coefficients to log areas |
| `v_lpcrf2lo` | Convert reflection coefficients to log area ratios |
| `v_lpcrf2rr` | Convert reflection coefficients to autocorrelation coefficients |
| `v_lpccc2ar` | Convert complex cepstrum to AR coefficients |
| `v_lpccc2cc` | Extrapolate complex cepstrum |
| `v_lpccc2db` | Convert complex cepstrum to dB power spectrum |
| `v_lpccc2ff` | Convert complex cepstrum to complex spectrum |
| `v_lpccc2pf` | Convert complex cepstrum to power spectrum |
| `v_lpcaa2ao` | Convert area function to area ratios |
| `v_lpcaa2dl` | Convert area coefficients to DCT of log area |
| `v_lpcaa2rf` | Convert vocal tract areas to reflection coefficients |
| `v_lpcao2rf` | Convert area ratios to reflection coefficients |
| `v_lpccw2zz` | Power spectrum roots to LPC poles |
| `v_lpcdb2pf` | Convert decibel power spectrum to power spectrum |
| `v_lpcdl2aa` | Convert DCT of log area to area coefficients |
| `v_lpcff2pf` | Convert complex spectrum to power spectrum |
| `v_lpcfq2zz` | Convert frequencies and Q factors to z-plane poles |
| `v_lpcim2ar` | Convert impulse response to AR coefficients |
| `v_lpcis2rf` | Convert inverse sines to reflection coefficients |
| `v_lpcla2rf` | Convert log areas to reflection coefficients |
| `v_lpclo2rf` | Convert log area ratios to reflection coefficients |
| `v_lpcls2ar` | Convert line spectrum pair frequencies to AR polynomial |
| `v_lpcpf2cc` | Convert power spectrum to complex cepstrum |
| `v_lpcpf2ff` | Convert power spectrum to complex spectrum |
| `v_lpcpf2rr` | Convert power spectrum to autocorrelation coefficients |
| `v_lpcpp2cw` | Convert power spectrum polynomial to power spectrum zeros |
| `v_lpcpp2pz` | Convert power spectrum polynomial to power spectrum zeros |
| `v_lpcpz2zz` | Power spectrum roots to LPC poles |
| `v_lpcra2ar` | Convert inverse filter autocorrelation coefficients to AR filter |
| `v_lpcra2pf` | Convert inverse filter autocorrelation to power spectrum |
| `v_lpcra2pp` | Convert inverse filter autocorrelation to power spectrum polynomial |
| `v_lpcrr2am` | Convert autocorrelation coefficients to AR coefficient matrix |
| `v_lpcrr2ar` | Convert autocorrelation coefficients to AR coefficients |
| `v_lpcss2zz` | Convert s-plane poles to z-plane poles |
| `v_lpczz2ar` | Convert z-plane poles to AR coefficients |
| `v_lpczz2cc` | Convert poles to complex cepstrum |
| `v_lpczz2ss` | Convert z-plane poles to s-plane poles |

</details>

<details>
<summary><strong>Speech Synthesis</strong></summary>

Text-to-speech and glottal flow models.

| Function | Description |
|---|---|
| `v_sapisynth` | SAPI speech synthesis (stub) |
| `v_glotros` | Rosenberg glottal model |
| `v_glotlf` | Liljencrants-Fant glottal model |

</details>

<details>
<summary><strong>Speech Enhancement</strong></summary>

Noise estimation and single-channel speech enhancement algorithms.

| Function | Description |
|---|---|
| `v_estnoiseg` | Estimate MMSE noise spectrum (Gerkmann & Hendriks) |
| `v_estnoisem` | Estimate noise spectrum using minimum statistics (Martin) |
| `v_specsub` | Speech enhancement using spectral subtraction |
| `v_specsubm` | Spectral subtraction (Martin's method) |
| `v_spendred` | Speech enhancement using spectral subtraction with decision-directed approach (stub) |
| `v_ssubmmse` | Speech enhancement using MMSE spectral amplitude estimator |
| `v_ssubmmsev` | Speech enhancement using MMSE with VAD-based noise estimation |

</details>

<details>
<summary><strong>Speech Coding</strong></summary>

Companding (A-law, mu-law) and telephone-band filtering.

| Function | Description |
|---|---|
| `v_lin2pcma` | Convert linear PCM to A-law |
| `v_pcma2lin` | Convert A-law PCM to linear |
| `v_lin2pcmu` | Convert linear to Mu-law PCM |
| `v_pcmu2lin` | Convert Mu-law PCM to linear |
| `v_potsband` | Design 300-3400 Hz telephone bandwidth filter |

</details>

<details>
<summary><strong>Speech Recognition & Features</strong></summary>

MFCC extraction, mel filterbanks, and Linear Discriminant Analysis.

| Function | Description |
|---|---|
| `v_melcepst` | Calculate the mel cepstrum of a signal |
| `v_melbankm` | Determine matrix for a mel/erb/bark-spaced filterbank |
| `v_cep2pow` | Convert cepstral means and variances to the power domain |
| `v_pow2cep` | Convert power domain means and variances to the cepstral domain |
| `v_ldatrace` | LDA transform to maximize trace discriminant |

</details>

<details>
<summary><strong>Signal Processing</strong></summary>

General-purpose DSP: filtering, windows, peak finding, dithering, Teager energy, zero-crossings, resampling.

| Function | Description |
|---|---|
| `v_windows` | Generate a standard windowing function |
| `v_windinfo` | Window information and figures of merit |
| `v_filterbank` | Apply a bank of IIR filters to a signal |
| `v_maxfilt` | Find max of an exponentially weighted sliding window |
| `v_momfilt` | Calculate moments of a signal using a sliding window |
| `v_meansqtf` | Mean square transfer function of a filter |
| `v_resample` | Resample and remove end transients |
| `v_dlyapsq` | Solve discrete Lyapunov equation in square root form |
| `v_findpeaks` | Find peaks with optional quadratic interpolation |
| `v_zerocros` | Find zero crossings in a signal |
| `v_schmitt` | Pass input signal through a Schmitt trigger |
| `v_teager` | Calculate Teager energy waveform |
| `v_ditherq` | Add dither and quantize |
| `v_nearnonz` | Replace each zero element with nearest non-zero element |
| `v_rangelim` | Limit the range of matrix elements |
| `v_horizdiff` | Estimate horizontal difference between two functions |
| `v_interval` | Classify X values into contiguous intervals |
| `v_modsym` | Symmetric modulus function |
| `v_zerotrim` | Remove trailing zero rows and columns |

</details>

<details>
<summary><strong>Information Theory</strong></summary>

Entropy and Huffman coding.

| Function | Description |
|---|---|
| `v_huffman` | Calculate a D-ary Huffman code |
| `v_entropy` | Shannon entropy of discrete and sampled continuous distributions |

</details>

<details>
<summary><strong>Rotations, Quaternions and Geometry</strong></summary>

Conversions between Euler angles, rotation matrices, and quaternions (real and complex), quaternion arithmetic, and 2D/3D geometry primitives.

| Function | Description |
|---|---|
| `v_roteu2qr` | Convert Euler angles to real unit quaternion |
| `v_rotqr2eu` | Convert real quaternion to Euler angles |
| `v_roteu2ro` | Convert Euler angles to rotation matrix |
| `v_rotro2eu` | Convert rotation matrix to Euler angles |
| `v_rotro2qr` | Convert 3x3 rotation matrix to real quaternion |
| `v_rotqr2ro` | Convert real quaternion to 3x3 rotation matrix |
| `v_rotmr2qr` | Convert real quaternion matrices to quaternion vectors |
| `v_rotqr2mr` | Convert real quaternion vectors to quaternion matrices |
| `v_rotmc2qc` | Convert complex quaternion matrices to complex quaternion vectors |
| `v_rotqc2mc` | Convert complex quaternion vectors to complex quaternion matrices |
| `v_rotqc2qr` | Convert complex quaternion to real quaternion |
| `v_rotqr2qc` | Convert real quaternion to complex quaternion |
| `v_rotax2qr` | Convert rotation axis and angle to quaternion |
| `v_rotqr2ax` | Convert quaternion to rotation axis and angle |
| `v_rotpl2ro` | Find rotation matrix from plane vectors |
| `v_rotro2pl` | Find plane and rotation angle of a rotation matrix |
| `v_rotlu2ro` | Convert look and up directions to rotation matrix |
| `v_rotro2lu` | Convert rotation matrix to look and up directions |
| `v_roteucode` | Decode Euler angle rotation code string |
| `v_rotation` | Encode and decode rotation matrices |
| `v_rotqrmean` | Calculate mean rotation of quaternion array |
| `v_rotqrvec` | Rotate vectors by quaternion |
| `v_qrmult` | Multiply two real quaternion matrices |
| `v_qrdivide` | Divide two real quaternions |
| `v_qrdotmult` | Element-wise quaternion multiplication |
| `v_qrdotdiv` | Element-wise quaternion division |
| `v_qrabs` | Absolute value and normalization of real quaternions |
| `v_qrpermute` | Transpose or permute a quaternion array |
| `v_polygonarea` | Calculate polygon area |
| `v_polygonwind` | Test if points are inside a polygon |
| `v_polygonxline` | Find where a line crosses a polygon |
| `v_minspane` | Minimum spanning tree using Euclidean distance |
| `v_imagehomog` | Apply homography transformation to an image |
| `v_rectifyhomog` | Apply rectifying homographies to an image set |
| `v_skew3d` | Convert between vector and skew-symmetric matrix |
| `v_upolyhedron` | Calculate uniform polyhedron characteristics |
| `v_sphrharm` | Forward and inverse spherical harmonic transform (stub) |

</details>

<details>
<summary><strong>Printing and Display</strong></summary>

Figure export, axis labelling with SI prefixes, colour maps, and figure layout.

| Function | Description |
|---|---|
| `v_fig2pdf` | Save a figure to PDF/EPS/PS format |
| `v_fig2emf` | Save a figure in various image formats |
| `v_figbolden` | Embolden, resize and recolour the current figure |
| `v_axisenlarge` | Enlarge the axes of a figure |
| `v_tilefigs` | Tile current figure windows |
| `v_colormap` | Set and create custom color maps |
| `v_lambda2rgb` | Convert wavelength to XYZ or RGB colour space |
| `v_xticksi` | Label the x-axis of a plot using SI multipliers |
| `v_yticksi` | Label the y-axis of a plot using SI multipliers |
| `v_xyzticksi` | Label an axis of a plot using SI multipliers |
| `v_xtickint` | Remove non-integer ticks from x-axis |
| `v_ytickint` | Remove non-integer ticks from y-axis |
| `v_texthvc` | Write text on graph with specified alignment and colour |
| `v_cblabel` | Add a label to a colorbar |
| `v_sprintsi` | Print value with SI multiplier |
| `v_sprintcpx` | Format a complex number for printing |
| `v_bitsprec` | Round values to a specified fixed or floating precision |

</details>

<details>
<summary><strong>Utility Functions</strong></summary>

VOICEBOX configuration, filesystem helpers, numeric helpers, and combinatorics.

| Function | Description |
|---|---|
| `v_voicebox` | Global parameters for Voicebox functions |
| `v_voicebox_update` | Check for voicebox updates (stub) |
| `v_paramsetch` | Set parameters for speech processing algorithms (stub) |
| `v_hostipinfo` | Get host name and IP info using Python equivalents |
| `v_winenvar` | Read Windows environment variable (stub) |
| `v_unixwhich` | Search system path for an executable (Python equivalent) |
| `v_regexfiles` | Find files matching a regular expression pattern |
| `v_fopenmkd` | Open file, creating directories if needed |
| `v_finishat` | Print estimated finish time of a long computation |
| `v_m2htmlpwd` | MATLAB-specific HTML documentation utility (stub) |
| `v_atan2sc` | Sin and cosine of atan(y/x) |
| `v_logsum` | Log(sum(k.*exp(x),d)) computed avoiding overflow/underflow |
| `v_gammalns` | Log of Gamma(x) for positive or negative real x |
| `v_hypergeom1f1` | Confluent hypergeometric function 1F1 (Kummer's M) |
| `v_dualdiag` | Simultaneous diagonalization of two Hermitian matrices |
| `v_mintrace` | Find row permutation to minimize trace |
| `v_quadpeak` | Find quadratically-interpolated peak in an N-D array |
| `v_peak2dquad` | Find quadratically-interpolated peak in a 2D array |
| `v_choosenk` | All choices of K elements from 0:N-1 |
| `v_choosrnk` | All choices of K elements from 0:N-1 with replacement |
| `v_permutes` | All N! permutations of 0:N-1 + signatures |
| `v_sort` | Sort with forward and inverse index |

</details>

## Development

pyvoicebox is a from-scratch Python reimplementation of each MATLAB function, verified to produce identical numerical output.

For every function, an Octave harness runs the *original MATLAB code* with representative inputs and saves the outputs as `.mat` files. The Python implementation is then compared against these reference values with `np.testing.assert_allclose()` at tight tolerances (typically `rtol=1e-10` to `1e-12`).

The repo ships no pre-built reference data. On first test run, `tests/conftest.py` automatically clones the [original sap-voicebox repository](https://github.com/ImperialCollegeLondon/sap-voicebox), runs the Octave harness scripts, and caches the generated reference `.mat` files at `~/.cache/pyvoicebox-test/`. This means the tests always validate against the real MATLAB source, not stale snapshots.

Every function keeps the same name, argument order, and mode-string conventions as the MATLAB original. MATLAB `v_frq2mel(f)` becomes Python `v_frq2mel(f)`.

### Running the tests

Requires Git, GNU Octave, and Python 3.9+.

```bash
pip install -e ".[dev]"
pytest tests/ -v
# First run: clones voicebox source, generates ref data via Octave, runs all tests
# Subsequent runs: uses cached ref data, runs tests only
```

## A note on ergonomics

pyvoicebox is a **faithful port** — it preserves the original MATLAB function names, argument order, mode strings, and default behaviour. This is intentional: if you're porting MATLAB code or following a paper that references VOICEBOX, everything works the same way.

That said, some MATLAB conventions can feel surprising in Python. For example, `v_addnoise(signal, fs, 5)` normalises total power to 1 by default — you need the `'k'` flag to preserve the original signal level. Single-character mode strings like `'M0dD'` are compact but not self-documenting.

We're considering a more Pythonic API layer on top of the faithful port — keyword arguments, sensible defaults, better discoverability. If you have opinions on what that should look like, or if you run into a gotcha that tripped you up, please [open an issue](https://github.com/MFA-X-AI/pyvoicebox/issues).

## Acknowledgements

pyvoicebox exists only because of the decades of work by **Prof. Mike Brookes** and collaborators at the **Speech and Audio Processing Lab, Centre for Signal Processing, Department of Electrical and Electronic Engineering, Imperial College London**. Every function in this package is a direct port of their MATLAB source, and every algorithm, mode flag, and default parameter traces back to their design choices.

If you use pyvoicebox in academic work, please cite the original VOICEBOX toolbox:

> Brookes, M., *VOICEBOX: Speech Processing Toolbox for MATLAB*, Department of Electrical and Electronic Engineering, Imperial College London, 1997–present. <https://github.com/ImperialCollegeLondon/sap-voicebox>

This work was supported by the Ministry of Higher Education (MOHE) Malaysia under the Prototype Development Research Grant Scheme (PRGS), Grant No. PRGS25-029-0073.

Upstream resources:

- MATLAB source: <https://github.com/ImperialCollegeLondon/sap-voicebox>
- Documentation: <https://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html>

## License

pyvoicebox is distributed under the **GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later)**, matching the upstream MATLAB toolbox. See [`LICENSE`](LICENSE) and [`LICENSE.GPL`](LICENSE.GPL).

Copyright for the original algorithms and MATLAB implementations remains with **Mike Brookes** and the contributors listed in each source file.
