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

## Installation

```bash
pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy`, `soundfile`, `matplotlib`. Tests additionally require `pytest`.

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

## Function Reference

Functions are grouped by topic, following the same categorisation as the [original VOICEBOX documentation](https://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html).

### Audio File Input/Output

Read and write a variety of audio formats: WAV, HTK, SPHERE/TIMIT, AIFF, AU, FLAC, Connex, SFS.

`v_readwav` / `v_writewav`, `v_readhtk` / `v_writehtk`, `v_readsph`, `v_readaif`, `v_readau`, `v_readflac`, `v_readsfs`, `v_readcnx`

### Frequency Scale Conversion

Convert between Hz and perceptual scales (Mel, Bark, ERB) as well as musical scales (Cent, MIDI).

`v_frq2mel` / `v_mel2frq`, `v_frq2bark` / `v_bark2frq`, `v_frq2erb` / `v_erb2frq`, `v_frq2cent` / `v_cent2frq`, `v_frq2midi` / `v_midi2frq`

### Fourier, DCT and Hartley Transforms

FFTs on real data (only returning the non-redundant half), DCT, Hartley, zoom FFT, FFT-based convolution.

`v_rfft` / `v_irfft`, `v_rsfft`, `v_zoomfft`, `v_rdct` / `v_irdct`, `v_rhartley`, `v_convfft`, `v_frac2bin`

### Random Numbers and Probability Distributions

Random number generators, multivariate Gaussian mixtures (fit / score / merge / divergence / transform), k-means, and various probability densities.

*Random*: `v_randvec`, `v_randiscr`, `v_stdspectrum`, `v_randfilt`, `v_rnsubset`, `v_usasi`
*Gaussian mixtures*: `v_gaussmix`, `v_gaussmixb`, `v_gaussmixd`, `v_gaussmixg`, `v_gaussmixk`, `v_gaussmixm`, `v_gaussmixp`, `v_gaussmixt`, `v_gmmlpdf`, `v_kmeans`, `v_kmeanlbg`, `v_kmeanhar`
*Distributions*: `v_lognmpdf`, `v_normcdflog`, `v_chimv`, `v_vonmisespdf`, `v_maxgauss`, `v_berk2prob` / `v_prob2berk`, `v_gausprod`, `v_histndim`, `v_pdfmoments`, `v_besselratio`, `v_besselratioi`, `v_besratinv0`

### Vector Distance

Spectral distance measures between LPC filter pairs (Itakura, Itakura–Saito, COSH).

`v_disteusq`, `v_distitar`, `v_distitpf`, `v_distisar`, `v_distispf`, `v_distchar`, `v_distchpf`

### Speech Analysis

Frame-based analysis, spectrograms, pitch trackers, voice activity detection, psychoacoustic metrics, and SNR estimation.

*Framing & time-frequency*: `v_enframe`, `v_overlapadd`, `v_fram2wav`, `v_stftw` / `v_istftw`, `v_filtbankm`, `v_gammabank`, `v_spgrambw`, `v_modspect`, `v_correlogram`, `v_ewgrpdel`
*Pitch & voicing*: `v_fxpefac`, `v_fxrapt`, `v_dypsa`, `v_vadsohn`
*Level & noise*: `v_activlev`, `v_activlevg`, `v_earnoise`, `v_ppmvu`, `v_snrseg`, `v_addnoise`, `v_sigalign`, `v_txalign`
*Psychoacoustics*: `v_importsii`, `v_phon2sone` / `v_sone2phon`, `v_pesq2mos` / `v_mos2pesq`, `v_stoi2prob`, `v_psycdigit`, `v_psycest`, `v_psycestu`, `v_psychofunc`
*Misc*: `v_soundspeed`, `v_sigma`

### LPC Analysis of Speech

Autocorrelation / covariance LPC, bandwidth expansion, inverse filtering, stability enforcement, and a complete set of conversions between LPC representations (AR, reflection, line-spectra, cepstra, impulse response, power spectrum, poles/zeros, etc.).

*Core*: `v_lpcauto`, `v_lpccovar`, `v_lpcconv`, `v_lpcbwexp`, `v_lpcstable`, `v_lpcifilt`, `v_lpcrand`, `v_rootstab`, `v_ccwarpf`
*AR conversions (`v_lpcar2*`)*: `am`, `cc`, `db`, `ff`, `fm`, `im`, `ls`, `pf`, `pp`, `ra`, `rf`, `rr`, `zz`
*Reflection conversions (`v_lpcrf2*` / `*2rf`)*: `aa`, `ao`, `ar`, `is`, `la`, `lo`, `rr`
*Cepstral conversions (`v_lpccc2*`)*: `ar`, `cc`, `db`, `ff`, `pf`
*Other pair conversions*: `v_lpcaa2ao`, `v_lpcaa2dl`, `v_lpcaa2rf`, `v_lpcao2rf`, `v_lpccovar`, `v_lpccw2zz`, `v_lpcdb2pf`, `v_lpcdl2aa`, `v_lpcff2pf`, `v_lpcfq2zz`, `v_lpcim2ar`, `v_lpcls2ar`, `v_lpcpf2cc`, `v_lpcpf2ff`, `v_lpcpf2rr`, `v_lpcpp2cw`, `v_lpcpp2pz`, `v_lpcpz2zz`, `v_lpcra2ar`, `v_lpcra2pf`, `v_lpcra2pp`, `v_lpcrr2am`, `v_lpcrr2ar`, `v_lpcss2zz`, `v_lpczz2ar`, `v_lpczz2cc`, `v_lpczz2ss`

### Speech Synthesis

`v_sapisynth` (text-to-speech via Windows SAPI), `v_glotros` (Rosenberg glottal flow), `v_glotlf` (Liljencrants–Fant glottal flow)

### Speech Enhancement

Noise estimation and single-channel enhancement (spectral subtraction, MMSE).

`v_estnoiseg`, `v_estnoisem`, `v_specsub`, `v_specsubm`, `v_spendred`, `v_ssubmmse`, `v_ssubmmsev`

### Speech Coding

Companding (A-law / mu-law) and telephone-band filtering.

`v_lin2pcma` / `v_pcma2lin`, `v_lin2pcmu` / `v_pcmu2lin`, `v_potsband`

### Speech Recognition & Features

MFCC extraction, mel filterbank construction, cepstrum/power-spectrum conversion, LDA.

`v_melcepst`, `v_melbankm`, `v_cep2pow`, `v_pow2cep`, `v_ldatrace`

### Signal Processing

General-purpose DSP: filtering, windows, peak finding, dithering, Teager energy, zero-crossings, resampling.

`v_windows`, `v_windinfo`, `v_findpeaks`, `v_maxfilt`, `v_filterbank`, `v_resample`, `v_ditherq`, `v_teager`, `v_schmitt`, `v_zerocros`, `v_momfilt`, `v_meansqtf`, `v_dlyapsq`, `v_nearnonz`, `v_rangelim`, `v_horizdiff`, `v_interval`, `v_modsym`, `v_zerotrim`

### Information Theory

`v_huffman` (optimum D-ary symbol code), `v_entropy`

### Rotations, Quaternions and Geometry

Conversions between Euler angles, rotation matrices (real and complex), and quaternions (real and complex), along with quaternion arithmetic and 2D/3D geometry primitives.

*Rotation conversions*: `v_roteu2qr`, `v_rotqr2eu`, `v_roteu2ro`, `v_rotro2eu`, `v_rotro2qr`, `v_rotqr2ro`, `v_rotmr2qr`, `v_rotqr2mr`, `v_rotmc2qc`, `v_rotqc2mc`, `v_rotqc2qr`, `v_rotqr2qc`, `v_rotax2qr`, `v_rotqr2ax`, `v_rotpl2ro`, `v_rotro2pl`, `v_rotlu2ro`, `v_rotro2lu`, `v_roteucode`, `v_rotation`
*Quaternion operations*: `v_rotqrmean`, `v_rotqrvec`, `v_qrmult`, `v_qrdivide`, `v_qrdotmult`, `v_qrdotdiv`, `v_qrabs`, `v_qrpermute`
*Geometry*: `v_polygonarea`, `v_polygonwind`, `v_polygonxline`, `v_minspane`, `v_imagehomog`, `v_rectifyhomog`, `v_skew3d`, `v_upolyhedron`, `v_sphrharm`

### Printing and Display

Figure export, axis labelling with SI prefixes, colour maps, figure tiling.

`v_fig2pdf`, `v_fig2emf`, `v_figbolden`, `v_axisenlarge`, `v_colormap`, `v_lambda2rgb`, `v_sprintsi`, `v_sprintcpx`, `v_tilefigs`, `v_xticksi`, `v_yticksi`, `v_xyzticksi`, `v_xtickint`, `v_ytickint`, `v_texthvc`, `v_cblabel`, `v_bitsprec`

### Utility Functions

VOICEBOX configuration, combinatorics, log-domain arithmetic, and small numerical helpers.

*Configuration*: `v_voicebox`, `v_voicebox_update`, `v_paramsetch`, `v_hostipinfo`, `v_winenvar`, `v_unixwhich`, `v_regexfiles`, `v_fopenmkd`, `v_finishat`, `v_m2htmlpwd`
*Numeric helpers*: `v_atan2sc`, `v_logsum`, `v_gammalns`, `v_hypergeom1f1`, `v_dualdiag`, `v_mintrace`, `v_quadpeak`, `v_peak2dquad`
*Combinatorics*: `v_choosenk`, `v_choosrnk`, `v_permutes`, `v_sort`

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
# First run: clones voicebox source, generates ref data via Octave, runs 511 tests
# Subsequent runs: uses cached ref data, runs tests only
```

## Origin

The original VOICEBOX toolbox is maintained by Mike Brookes at the Speech and Audio Processing Lab, Centre for Signal Processing, Department of Electrical and Electronic Engineering, Imperial College London.

- MATLAB source: https://github.com/ImperialCollegeLondon/sap-voicebox
- Documentation: https://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html

## License

This port follows the licensing of the original VOICEBOX toolbox. See the [original repository](https://github.com/ImperialCollegeLondon/sap-voicebox) for license details.
