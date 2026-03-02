# VOICEBOX MATLAB → Python Conversion Plan

## Context

The VOICEBOX library is a comprehensive MATLAB speech processing toolkit with 544 `.m` files. We're converting the entire library to Python, one `.py` file per function, using GNU Octave to generate reference test data for validation.

## Project Structure

```
sap-voicebox/
├── voicebox/                    # Original MATLAB (unchanged)
├── pyvoicebox/                  # New Python package
│   ├── pyproject.toml
│   ├── __init__.py              # Imports all public functions
│   ├── v_frq2mel.py             # One file per function
│   ├── v_mel2frq.py
│   ├── v_rfft.py
│   ├── ...                      # (all 282 v_* functions)
│   ├── _compat.py               # Shared MATLAB compat helpers
│   └── _voicebox_config.py      # Python equivalent of v_voicebox globals
├── tests/
│   ├── octave_harness/          # Octave scripts to generate reference data
│   │   ├── gen_ref_freq.m       # Generate reference for frequency functions
│   │   ├── gen_ref_lpc.m        # Generate reference for LPC functions
│   │   └── ...
│   ├── ref_data/                # Saved reference inputs/outputs (.npz files)
│   ├── test_freq.py             # pytest tests for frequency conversion
│   ├── test_lpc.py              # pytest tests for LPC functions
│   ├── test_fft.py
│   └── ...
└── scripts/
    └── run_octave_refs.sh       # Script to run all Octave reference generators
```

## Dependencies

```
numpy
scipy
soundfile          # Audio I/O (replaces v_readwav/v_writewav)
matplotlib         # Plotting functions
pytest             # Testing
```

## Shared Compatibility Module (`_compat.py`)

Helpers for recurring MATLAB→Python patterns:
- `matlab_mod()` — MATLAB's mod behavior
- `atleast_col()` / `atleast_row()` — ensure column/row vectors
- `matlab_reshape()` — column-major reshape wrapper
- `persistent_cache` decorator — replaces MATLAB `persistent` variables

## Conversion Strategy: 10 Phases (dependency order)

### Phase 1: Infrastructure & Core Utilities (~15 functions)

Foundation everything else depends on.

| Function | File | Notes |
|----------|------|-------|
| `_compat.py` | new | MATLAB compatibility helpers |
| `_voicebox_config.py` | new | Global config (replaces v_voicebox) |
| `v_voicebox` | v_voicebox.m | Config/path management (44 callers) |
| `v_logsum` | v_logsum.m | Log-sum-exp (8 callers) |
| `v_gammalns` | v_gammalns.m | Signed log-gamma |
| `v_entropy` | v_entropy.m | Shannon entropy |
| `v_bitsprec` | v_bitsprec.m | Bits of precision |
| `v_fopenmkd` | v_fopenmkd.m | File open with mkdir |
| `v_finishat` | v_finishat.m | Progress timer |
| `v_rnsubset` | v_rnsubset.m | Random subset |
| `v_permutes` | v_permutes.m | Permutation utilities |
| `v_zerotrim` | v_zerotrim.m | Trim trailing zeros |
| `v_sort*` | v_sort*.m | Sort utilities |
| `v_windinfo` | v_windinfo.m | Window info |
| `v_windows` | v_windows.m | Window functions (15+ types) |

### Phase 2: Frequency Scale Conversions (10 functions)

All leaf functions, pure math.

| Function | Notes |
|----------|-------|
| `v_frq2mel` / `v_mel2frq` | Hz ↔ Mel |
| `v_frq2bark` / `v_bark2frq` | Hz ↔ Bark |
| `v_frq2erb` / `v_erb2frq` | Hz ↔ ERB |
| `v_frq2cent` / `v_cent2frq` | Hz ↔ Cent |
| `v_frq2midi` / `v_midi2frq` | Hz ↔ MIDI |

### Phase 3: FFT / Transform Operations (~10 functions)

Core signal processing building blocks.

| Function | Notes |
|----------|-------|
| `v_rfft` / `v_irfft` | Real FFT (28+10 callers) |
| `v_rsfft` | Real symmetric FFT |
| `v_rdct` / `v_irdct` | Real DCT |
| `v_rhartley` | Hartley transform |
| `v_zoomfft` | Zoom FFT |
| `v_sphrharm` | Spherical harmonics |
| `v_convfft` | FFT convolution |

### Phase 4: Signal Processing Primitives (~25 functions)

Frame-based processing, filtering, windowing.

| Function | Notes |
|----------|-------|
| `v_enframe` / `v_overlapadd` | Frame extraction/reconstruction |
| `v_fram2wav` | Frames to waveform |
| `v_ditherq` | Dithered quantization |
| `v_findpeaks` | Peak finding |
| `v_maxfilt` | Running max filter |
| `v_teager` | Teager energy |
| `v_zerocros` | Zero crossing detection |
| `v_resample` | Signal resampling |
| `v_schmitt` | Schmitt trigger |
| `v_addnoise` | Add noise at SNR |
| `v_randfilt` | Filtered random noise |
| `v_stdspectrum` | Standard spectrum (needs scipy for Signal Processing Toolbox equivalents) |
| `v_potsband` | Telephone band filter |
| `v_filterbank` / `v_filtbankm` | Filterbank matrices |
| `v_gammabank` | Gammatone filterbank |
| `v_ewgrpdel` | Group delay |
| `v_horizdiff` | Horizontal differentiation |
| `v_ccwarpf` | Cepstral warping |
| `v_stftw` / `v_istftw` | STFT and inverse |

### Phase 5: Audio I/O & Codecs (~18 functions)

File reading/writing and PCM codecs.

| Function | Notes |
|----------|-------|
| `v_lin2pcmu` / `v_pcmu2lin` | μ-law codec |
| `v_lin2pcma` / `v_pcma2lin` | A-law codec |
| `v_readwav` / `v_writewav` | WAV I/O (use soundfile internally) |
| `v_readhtk` / `v_writehtk` | HTK format |
| `v_readsfs` | SFS format |
| `v_readsph` | SPHERE/TIMIT format |
| `v_readaif` | AIFF format |
| `v_readau` | AU format |
| `v_readflac` | FLAC format |
| `v_readcnx` | CNX format |
| `v_activlev` / `v_activlevg` | Active speech level (ITU P.56) |
| `v_earnoise` | Ear noise |

### Phase 6: LPC Analysis (~61 functions)

Linear Predictive Coding — largest category.

Convert in sub-order:
1. **Core LPC analysis**: `v_lpcauto`, `v_lpccovar`, `v_lpcconv`, `v_lpcbwexp`, `v_lpcstable`, `v_lpcifilt`, `v_lpcrand`
2. **AR-based conversions** (v_lpcar2*): `ar2cc`, `ar2db`, `ar2ff`, `ar2fm`, `ar2im`, `ar2ls`, `ar2pf`, `ar2pp`, `ar2ra`, `ar2rf`, `ar2zz`, `ar2am`
3. **Other representation conversions** (v_lpc*2*): `aa2ao`, `aa2dl`, `aa2rf`, `ao2rf`, `cc2ar`, `cc2cc`, `cc2ff`, `cw2zz`, `db2pf`, `ff2pf`, `fq2zz`, `is2rf`, `la2rf`, `lo2rf`, `ls2ar`, `pp2cw`, `pp2pz`, `pz2zz`, `ra2ar`, `ra2pp`, `rf2ar`, `rf2la`, `rf2lo`, `rr2ar`, `zz2ar`, `zz2pf`
4. **Utilities**: `v_rootstab`, `v_lpccovar`

### Phase 7: Gaussian Mixture Models & Probability (~25 functions)

Statistical modeling.

| Function | Notes |
|----------|-------|
| `v_gaussmixp` | GMM probability (22 callers) |
| `v_gaussmix` | GMM fitting (EM algorithm) |
| `v_gaussmixb` | GMM batch |
| `v_gaussmixd` | GMM distribution |
| `v_gaussmixg` | GMM graphics |
| `v_gaussmixk` | GMM divergence |
| `v_gaussmixm` | GMM merge |
| `v_gaussmixt` | GMM transform |
| `v_gmmlpdf` | GMM log-PDF |
| `v_gausprod` | Gaussian product |
| `v_randvec` | Random vectors |
| `v_randiscr` | Discrete random |
| `v_kmeans` / `v_kmeanlbg` / `v_kmeanhar` | K-means variants |
| `v_lognmpdf` | Log-normal PDF |
| `v_maxgauss` | Max of Gaussian vector |
| `v_histndim` | N-dim histogram |
| `v_pdfmoments` | PDF moment conversion |
| `v_chimv` | Chi mean/variance |
| `v_berk2prob` | Berkson to probability |
| `v_vonmisespdf` | Von Mises distribution |

### Phase 8: Speech Analysis & Enhancement (~20 functions)

High-level speech processing.

| Function | Notes |
|----------|-------|
| `v_melbankm` | Mel filterbank matrix |
| `v_melcepst` | MFCC extraction |
| `v_cep2pow` / `v_pow2cep` | Cepstral ↔ power |
| `v_ldatrace` | LDA trace |
| `v_correlogram` | Correlogram |
| `v_spgrambw` | Spectrogram |
| `v_modspect` | Modulation spectrum |
| `v_estnoiseg` / `v_estnoisem` | Noise estimation |
| `v_specsub` / `v_specsubm` | Spectral subtraction |
| `v_ssubmmse` / `v_ssubmmsev` | MMSE enhancement |
| `v_spendred` | Enhancement + dereverberation |
| `v_dypsa` | Glottal closure detection |
| `v_fxpefac` | Pitch extraction (PEFAC) |
| `v_fxrapt` | Pitch extraction (RAPT) |
| `v_glotros` / `v_glotlf` | Glottal models |
| `v_sapisynth` | TTS synthesis |
| `v_importsii` | Speech Intelligibility Index |

### Phase 9: Rotation, Quaternion & Geometry (~25 functions)

3D math utilities.

All `v_rot*` functions:
- `v_roteu2qr`, `v_rotqr2eu` — Euler ↔ Quaternion
- `v_rotro2qr`, `v_rotqr2ro` — Rotation matrix ↔ Quaternion
- `v_rotmc2qr`, `v_rotqr2mc` — Matrix column ↔ Quaternion
- `v_rotqrmean`, `v_rotqrvec`, `v_rotqrmul` — Quaternion operations
- `v_rotro2eu` — Rotation matrix → Euler (uses v_rotro2eu_tab constants)
- Plus: `v_imagehomog`, `v_polygonarea`, `v_polygonwind`, `v_polygonxline`, `v_minspane`

### Phase 10: Plotting, Display & Legacy Wrappers (~275 functions)

Visualization + all unprefixed aliases.

**Plotting (~15 functions):**
- `v_figbolden`, `v_fig2pdf`, `v_fig2emf` — Figure export
- `v_axisenlarge` — Axis scaling (23 callers)
- `v_texthvc` — Text placement (19 callers)
- `v_cblabel` — Colorbar labels
- `v_xticksi` / `v_yticksi` — SI-prefix tick labels
- `v_tilefigs` — Tile figure windows
- `v_colormap` — Custom colormaps
- `v_lambda2rgb` — Wavelength to RGB

**Legacy wrappers (~260 files):**
Each is a thin wrapper that calls the `v_` prefixed version.
In Python: just re-export from `__init__.py` with aliases.

---

## Per-Function Conversion Workflow

For each function:

1. **Read** the MATLAB source, understand inputs/outputs/modes
2. **Write Octave test script** that exercises the function with representative inputs and saves results to `.mat`
3. **Run Octave** to generate reference data, convert `.mat` → `.npz`
4. **Write Python** equivalent using numpy/scipy
5. **Write pytest** that loads reference `.npz` and compares Python output
6. **Run tests**, fix discrepancies

## Testing & Verification

- **Octave reference data**: For each function, generate `.npz` files with input/output pairs
- **Numerical tolerance**: `np.allclose(python_out, octave_ref, rtol=1e-10, atol=1e-12)` for most functions
- **pytest**: `pytest tests/ -v` to run all comparisons
- **CI-friendly**: All reference data pre-generated and committed

## Key Technical Mappings

| MATLAB | Python |
|--------|--------|
| `persistent x` | Module-level `_cache = {}` or `functools.lru_cache` |
| `nargin < 3` | `def f(x, y, z=None)` with default args |
| `nargout > 1` | Always return tuple, let caller unpack |
| `x(:)` | `x.ravel(order='F')` |
| `x.'` | `x.T` |
| `x'` | `x.conj().T` |
| `[m,n] = size(x)` | `m, n = x.shape` |
| `zeros(m,n)` | `np.zeros((m, n))` |
| `x(1:end-1)` | `x[:-1]` (but watch 1-indexing!) |
| `struct.field` | `dict['field']` or dataclass |
| `cell array {}` | Python `list` |
| `sparse(...)` | `scipy.sparse.csr_matrix(...)` |
| `fft(x)` | `np.fft.fft(x)` |
| `fread(fid,...)` | `struct.unpack(...)` or `np.fromfile(...)` |
| `regexp(...)` | `re.search(...)` |
| `error('...')` | `raise ValueError('...')` |
| `warning('...')` | `warnings.warn('...')` |

## Estimated Scope

| Phase | Functions | Complexity |
|-------|-----------|------------|
| 1. Infrastructure | ~15 | Medium |
| 2. Frequency | 10 | Low |
| 3. FFT/Transform | ~10 | Medium |
| 4. Signal Processing | ~25 | Medium-High |
| 5. Audio I/O | ~18 | High |
| 6. LPC | ~61 | Medium (repetitive) |
| 7. GMM/Probability | ~25 | High |
| 8. Speech Analysis | ~20 | High |
| 9. Rotation/Geometry | ~25 | Medium |
| 10. Plotting + Wrappers | ~275 | Low (mostly wrappers) |
| **Total** | **~544** | |
