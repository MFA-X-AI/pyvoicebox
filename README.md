# pyvoicebox

A complete Python port of the [VOICEBOX](https://github.com/ImperialCollegeLondon/sap-voicebox) speech processing toolbox, originally written in MATLAB by Mike Brookes at Imperial College London's Speech and Audio Processing Lab.

**282 functions** ported across 10 phases, validated against the original MATLAB source via GNU Octave with **511 automated tests**.

## What is VOICEBOX?

VOICEBOX is a comprehensive MATLAB toolkit for speech and audio signal processing maintained since the 1990s. It covers areas that most Python audio libraries don't touch:

- **Linear Predictive Coding** -- 61 functions for LPC analysis and conversion between representations (AR coefficients, cepstra, reflection coefficients, line spectra, etc.)
- **Gaussian Mixture Models** -- Full GMM suite: fitting (EM), scoring, merging, divergence, batch processing
- **Speech Enhancement** -- Spectral subtraction, MMSE estimators, noise estimation (Rangachari & Loizou, Martin)
- **Pitch Detection** -- PEFAC (`v_fxpefac`), RAPT (`v_fxrapt`), DYPSA glottal closure detection
- **Psychoacoustics** -- PESQ/MOS mapping, Speech Intelligibility Index (SII), STOI, loudness (phon/sone)
- **Rotation & Quaternion Math** -- 30 functions for Euler angles, rotation matrices, quaternions, and geometry
- **Audio Codecs** -- mu-law, A-law, WAV, HTK, SPHERE/TIMIT, AIFF, AU, FLAC readers
- **Frequency Scales** -- Mel, Bark, ERB, Cent, MIDI conversions
- **Signal Processing** -- Enframing, overlap-add, STFT, filterbanks, zero-crossing detection, Teager energy

## Why not librosa?

librosa is excellent for music information retrieval (MIR) -- beat tracking, chroma features, harmonic/percussive separation. pyvoicebox targets a different domain:

| Area | pyvoicebox | librosa |
|------|-----------|---------|
| LPC analysis (61 representations) | Full suite | `lpc()` only |
| Gaussian mixtures | Complete (fit, score, merge, transform) | None |
| Speech enhancement | MMSE, spectral subtraction, noise estimation | None |
| Pitch detection | PEFAC, RAPT, DYPSA | `pyin` |
| Psychoacoustic metrics | PESQ, SII, STOI, phon/sone | None |
| Quaternion/rotation math | 30 functions | None |
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
cc = v_lpcar2cc(ar)                    # AR -> cepstral coefficients

# Quaternion operations
from pyvoicebox import v_roteu2qr, v_rotqr2ro
import numpy as np
q = v_roteu2qr('xyz', np.array([0.1, 0.2, 0.3]))  # Euler -> quaternion
R = v_rotqr2ro(q)                                     # quaternion -> rotation matrix

# Noise estimation & speech enhancement
from pyvoicebox import v_estnoiseg, v_specsub
```

All functions are available with both the `v_` prefix (matching MATLAB) and without (legacy aliases):

```python
from pyvoicebox import frq2mel       # same as v_frq2mel
from pyvoicebox import melcepst      # same as v_melcepst
from pyvoicebox import lpcauto       # same as v_lpcauto
```

## Function Reference

### Phase 1: Infrastructure & Core Utilities (16 functions)
`v_logsum`, `v_gammalns`, `v_entropy`, `v_bitsprec`, `v_windows`, `v_windinfo`, `v_voicebox`, `v_huffman`, `v_choosenk`, `v_choosrnk`, `v_permutes`, `v_sort`, `v_zerotrim`, `v_rnsubset`, `v_finishat`, `v_fopenmkd`

### Phase 2: Frequency Scale Conversions (10 functions)
`v_frq2mel` / `v_mel2frq`, `v_frq2bark` / `v_bark2frq`, `v_frq2erb` / `v_erb2frq`, `v_frq2cent` / `v_cent2frq`, `v_frq2midi` / `v_midi2frq`

### Phase 3: FFT & Transform Operations (9 functions)
`v_rfft` / `v_irfft`, `v_rdct` / `v_irdct`, `v_rsfft`, `v_rhartley`, `v_zoomfft`, `v_convfft`, `v_frac2bin`

### Phase 4: Signal Processing Primitives (22 functions)
`v_enframe`, `v_overlapadd`, `v_fram2wav`, `v_stftw` / `v_istftw`, `v_filtbankm`, `v_filterbank`, `v_findpeaks`, `v_maxfilt`, `v_ditherq`, `v_teager`, `v_zerocros`, `v_schmitt`, `v_sigalign`, `v_rangelim`, `v_nearnonz`, `v_stdspectrum`, `v_randfilt`, `v_resample`, `v_horizdiff`, `v_ccwarpf`, `v_interval`

### Phase 5: Audio I/O & Codecs (11 functions)
`v_readwav` / `v_writewav`, `v_readhtk` / `v_writehtk`, `v_lin2pcmu` / `v_pcmu2lin`, `v_lin2pcma` / `v_pcma2lin`, `v_activlev`, `v_activlevg`, `v_earnoise`

### Phase 6: LPC Analysis (61 functions)
Core: `v_lpcauto`, `v_lpccovar`, `v_lpcconv`, `v_lpcbwexp`, `v_lpcstable`, `v_lpcifilt`, `v_lpcrand`
AR conversions: `v_lpcar2cc`, `v_lpcar2db`, `v_lpcar2ff`, `v_lpcar2fm`, `v_lpcar2im`, `v_lpcar2ls`, `v_lpcar2pf`, `v_lpcar2pp`, `v_lpcar2ra`, `v_lpcar2rf`, `v_lpcar2rr`, `v_lpcar2zz`, `v_lpcar2am`
Other conversions: 40+ `v_lpc*2*` functions covering every LPC representation pair
Utilities: `v_rootstab`

### Phase 7: Gaussian Mixtures & Probability (28 functions)
`v_gaussmix`, `v_gaussmixb`, `v_gaussmixd`, `v_gaussmixg`, `v_gaussmixk`, `v_gaussmixm`, `v_gaussmixp`, `v_gaussmixt`, `v_gmmlpdf`, `v_gausprod`, `v_kmeans`, `v_kmeanlbg`, `v_kmeanhar`, `v_randvec`, `v_randiscr`, `v_disteusq`, `v_histndim`, `v_lognmpdf`, `v_maxgauss`, `v_normcdflog`, `v_pdfmoments`, `v_chimv`, `v_berk2prob` / `v_prob2berk`, `v_vonmisespdf`, `v_besselratio`, `v_besselratioi`, `v_besratinv0`

### Phase 8: Speech Analysis & Enhancement (29 functions)
`v_melcepst`, `v_melbankm`, `v_cep2pow` / `v_pow2cep`, `v_estnoiseg`, `v_estnoisem`, `v_specsub`, `v_specsubm`, `v_ssubmmse`, `v_ssubmmsev`, `v_spgrambw`, `v_modspect`, `v_correlogram`, `v_fxpefac`, `v_fxrapt`, `v_dypsa`, `v_glotlf`, `v_glotros`, `v_importsii`, `v_ldatrace`, `v_meansqtf`, `v_addnoise`, `v_snrseg`, `v_phon2sone` / `v_sone2phon`, `v_pesq2mos` / `v_mos2pesq`, `v_stoi2prob`, `v_vadsohn`

### Phase 9: Rotation, Quaternion & Geometry (30 functions)
Euler/Quaternion/Matrix conversions: `v_roteu2qr`, `v_rotqr2eu`, `v_roteu2ro`, `v_rotro2eu`, `v_rotro2qr`, `v_rotqr2ro`, `v_rotmr2qr`, `v_rotqr2mr`, `v_rotmc2qc`, `v_rotqc2mc`, `v_rotqc2qr`, `v_rotqr2qc`, `v_rotax2qr`, `v_rotqr2ax`, `v_rotpl2ro`, `v_rotro2pl`, `v_rotlu2ro`, `v_rotro2lu`
Operations: `v_rotqrmean`, `v_rotqrvec`, `v_roteucode`, `v_rotation`
Geometry: `v_polygonarea`, `v_polygonwind`, `v_polygonxline`, `v_minspane`, `v_imagehomog`, `v_rectifyhomog`, `v_skew3d`, `v_upolyhedron`

### Phase 10 & 10b: Distance, Plotting & File Readers (65 functions)
Distance measures: `v_distchar`, `v_distchpf`, `v_distisar`, `v_distispf`, `v_distitar`, `v_distitpf`
Quaternion arithmetic: `v_qrmult`, `v_qrdivide`, `v_qrdotmult`, `v_qrdotdiv`, `v_qrabs`, `v_qrpermute`
Plotting: `v_fig2pdf`, `v_fig2emf`, `v_figbolden`, `v_axisenlarge`, `v_colormap`, `v_lambda2rgb`, `v_sprintsi`, `v_sprintcpx`, `v_tilefigs`, `v_xticksi`, `v_yticksi`, `v_xyzticksi`, `v_texthvc`, `v_cblabel`
File readers: `v_readaif`, `v_readau`, `v_readcnx`, `v_readflac`, `v_readsfs`, `v_readsph`
Misc: `v_soundspeed`, `v_quadpeak`, `v_peak2dquad`, `v_hypergeom1f1`, `v_dlyapsq`, `v_mintrace`, `v_ewgrpdel`, `v_txalign`, `v_modsym`, `v_sigma`, `v_potsband`, and more

## How It Was Built

pyvoicebox is a from-scratch Python reimplementation of each MATLAB function, verified to produce identical numerical output. It was built in 10 dependency-ordered phases:

1. **Phase-ordered porting** -- Functions were reimplemented in dependency order (infrastructure first, then frequency conversions, FFT, signal processing, etc.) so each phase could build on tested foundations.

2. **Octave-verified correctness** -- For each function, an Octave harness script (`tests/octave_harness/gen_ref_phase*.m`) runs the *original MATLAB code* with representative inputs and saves the outputs as `.mat` files. The Python implementation is then tested against these reference values, ensuring it matches the real MATLAB behaviour -- not just a best-guess reimplementation.

3. **Numerical validation** -- Each Python function is compared against the Octave reference output using `np.testing.assert_allclose()` with tight tolerances (typically `rtol=1e-10` to `1e-12`). The 511 tests cover edge cases, multiple input shapes, and all function modes.

4. **Self-verifying repo** -- The repo ships no pre-built reference data. On first test run, `tests/conftest.py` automatically:
   - Clones the [original sap-voicebox repository](https://github.com/ImperialCollegeLondon/sap-voicebox)
   - Runs all 11 Octave harness scripts to generate reference `.mat` files
   - Caches everything at `~/.cache/pyvoicebox-test/` for subsequent runs

   This means the tests always validate against the real MATLAB source, not stale snapshots.

5. **API preservation** -- Every function keeps the same name, argument order, and mode-string conventions as the MATLAB original. MATLAB `v_frq2mel(f)` becomes Python `v_frq2mel(f)`. Both `v_`-prefixed and unprefixed names are available.

### Prerequisites for running tests

- **Git** (to clone MATLAB source)
- **GNU Octave** (to generate reference data)
- **Python 3.9+** with numpy, scipy, soundfile, matplotlib

```bash
pip install -e ".[dev]"
pytest tests/ -v
# First run: clones voicebox source, generates ref data via Octave, runs 511 tests
# Subsequent runs: uses cached ref data, runs tests only
```

## Origin

The original VOICEBOX toolbox is maintained by Mike Brookes at the Speech and Audio Processing Lab, Centre for Signal Processing, Department of Electrical and Electronic Engineering, Imperial College London.

- MATLAB source: https://github.com/ImperialCollegeLondon/sap-voicebox
- Documentation: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html

## License

This port follows the licensing of the original VOICEBOX toolbox. See the [original repository](https://github.com/ImperialCollegeLondon/sap-voicebox) for license details.
