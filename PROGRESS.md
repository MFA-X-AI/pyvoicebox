# VOICEBOX MATLAB → Python Conversion Progress

> **Total: 0 / 282 core functions converted** (0%)
> Legacy wrappers: 0 / 258 (auto-generated once core functions are done)
> Demos: 0 / 3 | Constants: 0 / 1

---

## Phase 1: Infrastructure & Core Utilities (0 / 18)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 1 | `_compat.py` | n/a | [ ] | [ ] | Shared MATLAB compat helpers |
| 2 | `_voicebox_config.py` | n/a | [ ] | [ ] | Global config module |
| 3 | `v_voicebox` | [ ] | [ ] | [ ] | Config/path management (44 callers) |
| 4 | `v_logsum` | [ ] | [ ] | [ ] | Log-sum-exp |
| 5 | `v_gammalns` | [ ] | [ ] | [ ] | Signed log-gamma |
| 6 | `v_entropy` | [ ] | [ ] | [ ] | Shannon entropy |
| 7 | `v_bitsprec` | [ ] | [ ] | [ ] | Bits of precision |
| 8 | `v_fopenmkd` | [ ] | [ ] | [ ] | File open with mkdir |
| 9 | `v_finishat` | [ ] | [ ] | [ ] | Progress timer |
| 10 | `v_rnsubset` | [ ] | [ ] | [ ] | Random subset |
| 11 | `v_permutes` | [ ] | [ ] | [ ] | Permutation utilities |
| 12 | `v_zerotrim` | [ ] | [ ] | [ ] | Trim trailing zeros |
| 13 | `v_sort` | [ ] | [ ] | [ ] | Sort utility |
| 14 | `v_windinfo` | [ ] | [ ] | [ ] | Window info |
| 15 | `v_windows` | [ ] | [ ] | [ ] | Window functions (15+ types) |
| 16 | `v_choosenk` | [ ] | [ ] | [ ] | Choose N from K |
| 17 | `v_choosrnk` | [ ] | [ ] | [ ] | Random N from K |
| 18 | `v_huffman` | [ ] | [ ] | [ ] | Huffman coding |

---

## Phase 2: Frequency Scale Conversions (0 / 10)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 19 | `v_frq2mel` | [ ] | [ ] | [ ] | Hz → Mel |
| 20 | `v_mel2frq` | [ ] | [ ] | [ ] | Mel → Hz |
| 21 | `v_frq2bark` | [ ] | [ ] | [ ] | Hz → Bark |
| 22 | `v_bark2frq` | [ ] | [ ] | [ ] | Bark → Hz |
| 23 | `v_frq2erb` | [ ] | [ ] | [ ] | Hz → ERB-rate |
| 24 | `v_erb2frq` | [ ] | [ ] | [ ] | ERB-rate → Hz |
| 25 | `v_frq2cent` | [ ] | [ ] | [ ] | Hz → Cent |
| 26 | `v_cent2frq` | [ ] | [ ] | [ ] | Cent → Hz |
| 27 | `v_frq2midi` | [ ] | [ ] | [ ] | Hz → MIDI |
| 28 | `v_midi2frq` | [ ] | [ ] | [ ] | MIDI → Hz |

---

## Phase 3: FFT / Transform Operations (0 / 10)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 29 | `v_rfft` | [ ] | [ ] | [ ] | Real FFT (28 callers) |
| 30 | `v_irfft` | [ ] | [ ] | [ ] | Inverse real FFT (10 callers) |
| 31 | `v_rsfft` | [ ] | [ ] | [ ] | Real symmetric FFT |
| 32 | `v_rdct` | [ ] | [ ] | [ ] | Real DCT |
| 33 | `v_irdct` | [ ] | [ ] | [ ] | Inverse real DCT |
| 34 | `v_rhartley` | [ ] | [ ] | [ ] | Hartley transform |
| 35 | `v_zoomfft` | [ ] | [ ] | [ ] | Zoom FFT |
| 36 | `v_sphrharm` | [ ] | [ ] | [ ] | Spherical harmonics |
| 37 | `v_convfft` | [ ] | [ ] | [ ] | FFT convolution |
| 38 | `v_frac2bin` | [ ] | [ ] | [ ] | Fraction to binary |

---

## Phase 4: Signal Processing Primitives (0 / 30)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 39 | `v_enframe` | [ ] | [ ] | [ ] | Frame extraction (12 callers) |
| 40 | `v_overlapadd` | [ ] | [ ] | [ ] | Overlap-add reconstruction |
| 41 | `v_fram2wav` | [ ] | [ ] | [ ] | Frames → waveform |
| 42 | `v_ditherq` | [ ] | [ ] | [ ] | Dithered quantization |
| 43 | `v_findpeaks` | [ ] | [ ] | [ ] | Peak detection |
| 44 | `v_maxfilt` | [ ] | [ ] | [ ] | Running max filter |
| 45 | `v_teager` | [ ] | [ ] | [ ] | Teager energy operator |
| 46 | `v_zerocros` | [ ] | [ ] | [ ] | Zero crossing detection |
| 47 | `v_resample` | [ ] | [ ] | [ ] | Signal resampling |
| 48 | `v_schmitt` | [ ] | [ ] | [ ] | Schmitt trigger |
| 49 | `v_addnoise` | [ ] | [ ] | [ ] | Add noise at SNR |
| 50 | `v_randfilt` | [ ] | [ ] | [ ] | Filtered random noise |
| 51 | `v_stdspectrum` | [ ] | [ ] | [ ] | Standard spectrum (needs scipy) |
| 52 | `v_potsband` | [ ] | [ ] | [ ] | Telephone band filter |
| 53 | `v_filtbankm` | [ ] | [ ] | [ ] | Filterbank matrix |
| 54 | `v_filterbank` | [ ] | [ ] | [ ] | Filterbank |
| 55 | `v_gammabank` | [ ] | [ ] | [ ] | Gammatone filterbank |
| 56 | `v_ewgrpdel` | [ ] | [ ] | [ ] | Group delay |
| 57 | `v_horizdiff` | [ ] | [ ] | [ ] | Horizontal differentiation |
| 58 | `v_ccwarpf` | [ ] | [ ] | [ ] | Cepstral warping |
| 59 | `v_stftw` | [ ] | [ ] | [ ] | Short-time Fourier transform |
| 60 | `v_istftw` | [ ] | [ ] | [ ] | Inverse STFT |
| 61 | `v_momfilt` | [ ] | [ ] | [ ] | Moment filter |
| 62 | `v_meansqtf` | [ ] | [ ] | [ ] | Mean square transfer function |
| 63 | `v_sigalign` | [ ] | [ ] | [ ] | Signal alignment |
| 64 | `v_txalign` | [ ] | [ ] | [ ] | Text alignment |
| 65 | `v_paramsetch` | [ ] | [ ] | [ ] | Parameter set/check |
| 66 | `v_nearnonz` | [ ] | [ ] | [ ] | Nearest nonzero |
| 67 | `v_rangelim` | [ ] | [ ] | [ ] | Range limiting |
| 68 | `v_interval` | [ ] | [ ] | [ ] | Interval operations |

---

## Phase 5: Audio I/O & Codecs (0 / 20)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 69 | `v_lin2pcmu` | [ ] | [ ] | [ ] | Linear → μ-law |
| 70 | `v_pcmu2lin` | [ ] | [ ] | [ ] | μ-law → linear |
| 71 | `v_lin2pcma` | [ ] | [ ] | [ ] | Linear → A-law |
| 72 | `v_pcma2lin` | [ ] | [ ] | [ ] | A-law → linear |
| 73 | `v_readwav` | [ ] | [ ] | [ ] | Read WAV file |
| 74 | `v_writewav` | [ ] | [ ] | [ ] | Write WAV file |
| 75 | `v_readhtk` | [ ] | [ ] | [ ] | Read HTK format |
| 76 | `v_writehtk` | [ ] | [ ] | [ ] | Write HTK format |
| 77 | `v_readsfs` | [ ] | [ ] | [ ] | Read SFS format |
| 78 | `v_readsph` | [ ] | [ ] | [ ] | Read SPHERE/TIMIT |
| 79 | `v_readaif` | [ ] | [ ] | [ ] | Read AIFF format |
| 80 | `v_readau` | [ ] | [ ] | [ ] | Read AU format |
| 81 | `v_readflac` | [ ] | [ ] | [ ] | Read FLAC format |
| 82 | `v_readcnx` | [ ] | [ ] | [ ] | Read CNX format |
| 83 | `v_activlev` | [ ] | [ ] | [ ] | Active speech level (ITU P.56) |
| 84 | `v_activlevg` | [ ] | [ ] | [ ] | Active speech level (generic) |
| 85 | `v_earnoise` | [ ] | [ ] | [ ] | Ear noise model |
| 86 | `v_usasi` | [ ] | [ ] | [ ] | USASI noise spectrum |
| 87 | `v_snrseg` | [ ] | [ ] | [ ] | Segmental SNR |
| 88 | `v_ppmvu` | [ ] | [ ] | [ ] | PPM/VU meter |

---

## Phase 6: LPC Analysis (0 / 61)

### Core LPC (0 / 8)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 89 | `v_lpcauto` | [ ] | [ ] | [ ] | Autocorrelation LPC |
| 90 | `v_lpccovar` | [ ] | [ ] | [ ] | Covariance LPC |
| 91 | `v_lpcconv` | [ ] | [ ] | [ ] | LPC conversion utility |
| 92 | `v_lpcbwexp` | [ ] | [ ] | [ ] | Bandwidth expansion |
| 93 | `v_lpcstable` | [ ] | [ ] | [ ] | Stability check |
| 94 | `v_lpcifilt` | [ ] | [ ] | [ ] | Inverse filter |
| 95 | `v_lpcrand` | [ ] | [ ] | [ ] | Random LPC coefficients |
| 96 | `v_rootstab` | [ ] | [ ] | [ ] | Root stabilization |

### AR-based conversions — v_lpcar2* (0 / 13)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 97 | `v_lpcar2am` | [ ] | [ ] | [ ] | AR → area |
| 98 | `v_lpcar2cc` | [ ] | [ ] | [ ] | AR → cepstral |
| 99 | `v_lpcar2db` | [ ] | [ ] | [ ] | AR → dB spectrum |
| 100 | `v_lpcar2ff` | [ ] | [ ] | [ ] | AR → formant freq |
| 101 | `v_lpcar2fm` | [ ] | [ ] | [ ] | AR → formant (alt) |
| 102 | `v_lpcar2im` | [ ] | [ ] | [ ] | AR → impulse response |
| 103 | `v_lpcar2ls` | [ ] | [ ] | [ ] | AR → line spectral |
| 104 | `v_lpcar2pf` | [ ] | [ ] | [ ] | AR → power spectrum |
| 105 | `v_lpcar2pp` | [ ] | [ ] | [ ] | AR → pole positions |
| 106 | `v_lpcar2ra` | [ ] | [ ] | [ ] | AR → autocorrelation |
| 107 | `v_lpcar2rf` | [ ] | [ ] | [ ] | AR → reflection coeff |
| 108 | `v_lpcar2rr` | [ ] | [ ] | [ ] | AR → lattice |
| 109 | `v_lpcar2zz` | [ ] | [ ] | [ ] | AR → z-plane zeros |

### Other LPC conversions (0 / 40)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 110 | `v_lpcaa2ao` | [ ] | [ ] | [ ] | AA → AO |
| 111 | `v_lpcaa2dl` | [ ] | [ ] | [ ] | AA → DL |
| 112 | `v_lpcaa2rf` | [ ] | [ ] | [ ] | AA → RF |
| 113 | `v_lpcao2rf` | [ ] | [ ] | [ ] | AO → RF |
| 114 | `v_lpccc2ar` | [ ] | [ ] | [ ] | CC → AR |
| 115 | `v_lpccc2cc` | [ ] | [ ] | [ ] | CC → CC (resize) |
| 116 | `v_lpccc2db` | [ ] | [ ] | [ ] | CC → dB |
| 117 | `v_lpccc2ff` | [ ] | [ ] | [ ] | CC → formant freq |
| 118 | `v_lpccc2pf` | [ ] | [ ] | [ ] | CC → power spectrum |
| 119 | `v_lpccw2zz` | [ ] | [ ] | [ ] | CW → ZZ |
| 120 | `v_lpcdb2pf` | [ ] | [ ] | [ ] | DB → PF |
| 121 | `v_lpcdl2aa` | [ ] | [ ] | [ ] | DL → AA |
| 122 | `v_lpcff2pf` | [ ] | [ ] | [ ] | FF → PF |
| 123 | `v_lpcfq2zz` | [ ] | [ ] | [ ] | FQ → ZZ |
| 124 | `v_lpcim2ar` | [ ] | [ ] | [ ] | IM → AR |
| 125 | `v_lpcis2rf` | [ ] | [ ] | [ ] | IS → RF |
| 126 | `v_lpcla2rf` | [ ] | [ ] | [ ] | LA → RF |
| 127 | `v_lpclo2rf` | [ ] | [ ] | [ ] | LO → RF |
| 128 | `v_lpcls2ar` | [ ] | [ ] | [ ] | LS → AR |
| 129 | `v_lpcpf2cc` | [ ] | [ ] | [ ] | PF → CC |
| 130 | `v_lpcpf2ff` | [ ] | [ ] | [ ] | PF → FF |
| 131 | `v_lpcpf2rr` | [ ] | [ ] | [ ] | PF → RR |
| 132 | `v_lpcpp2cw` | [ ] | [ ] | [ ] | PP → CW |
| 133 | `v_lpcpp2pz` | [ ] | [ ] | [ ] | PP → PZ |
| 134 | `v_lpcpz2zz` | [ ] | [ ] | [ ] | PZ → ZZ |
| 135 | `v_lpcra2ar` | [ ] | [ ] | [ ] | RA → AR |
| 136 | `v_lpcra2pf` | [ ] | [ ] | [ ] | RA → PF |
| 137 | `v_lpcra2pp` | [ ] | [ ] | [ ] | RA → PP |
| 138 | `v_lpcrf2aa` | [ ] | [ ] | [ ] | RF → AA |
| 139 | `v_lpcrf2ao` | [ ] | [ ] | [ ] | RF → AO |
| 140 | `v_lpcrf2ar` | [ ] | [ ] | [ ] | RF → AR |
| 141 | `v_lpcrf2is` | [ ] | [ ] | [ ] | RF → IS |
| 142 | `v_lpcrf2la` | [ ] | [ ] | [ ] | RF → LA |
| 143 | `v_lpcrf2lo` | [ ] | [ ] | [ ] | RF → LO |
| 144 | `v_lpcrf2rr` | [ ] | [ ] | [ ] | RF → RR |
| 145 | `v_lpcrr2am` | [ ] | [ ] | [ ] | RR → AM |
| 146 | `v_lpcrr2ar` | [ ] | [ ] | [ ] | RR → AR |
| 147 | `v_lpcss2zz` | [ ] | [ ] | [ ] | SS → ZZ |
| 148 | `v_lpczz2ar` | [ ] | [ ] | [ ] | ZZ → AR |
| 149 | `v_lpczz2cc` | [ ] | [ ] | [ ] | ZZ → CC |
| 150 | `v_lpczz2ss` | [ ] | [ ] | [ ] | ZZ → SS |

---

## Phase 7: Gaussian Mixtures & Probability (0 / 27)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 151 | `v_gaussmixp` | [ ] | [ ] | [ ] | GMM probability (22 callers) |
| 152 | `v_gaussmix` | [ ] | [ ] | [ ] | GMM fitting (EM) |
| 153 | `v_gaussmixb` | [ ] | [ ] | [ ] | GMM batch update |
| 154 | `v_gaussmixd` | [ ] | [ ] | [ ] | GMM distribution |
| 155 | `v_gaussmixg` | [ ] | [ ] | [ ] | GMM graphics |
| 156 | `v_gaussmixk` | [ ] | [ ] | [ ] | GMM divergence |
| 157 | `v_gaussmixm` | [ ] | [ ] | [ ] | GMM merge |
| 158 | `v_gaussmixt` | [ ] | [ ] | [ ] | GMM transform |
| 159 | `v_gmmlpdf` | [ ] | [ ] | [ ] | GMM log-PDF |
| 160 | `v_gausprod` | [ ] | [ ] | [ ] | Gaussian product |
| 161 | `v_randvec` | [ ] | [ ] | [ ] | Random vectors |
| 162 | `v_randiscr` | [ ] | [ ] | [ ] | Discrete random values |
| 163 | `v_kmeans` | [ ] | [ ] | [ ] | K-means clustering |
| 164 | `v_kmeanlbg` | [ ] | [ ] | [ ] | K-means LBG variant |
| 165 | `v_kmeanhar` | [ ] | [ ] | [ ] | K-means harmonic |
| 166 | `v_lognmpdf` | [ ] | [ ] | [ ] | Log-normal PDF |
| 167 | `v_maxgauss` | [ ] | [ ] | [ ] | Max of Gaussian vector |
| 168 | `v_histndim` | [ ] | [ ] | [ ] | N-dimensional histogram |
| 169 | `v_pdfmoments` | [ ] | [ ] | [ ] | PDF moment conversion |
| 170 | `v_chimv` | [ ] | [ ] | [ ] | Chi mean/variance |
| 171 | `v_berk2prob` | [ ] | [ ] | [ ] | Berkson → probability |
| 172 | `v_prob2berk` | [ ] | [ ] | [ ] | Probability → Berkson |
| 173 | `v_vonmisespdf` | [ ] | [ ] | [ ] | Von Mises distribution |
| 174 | `v_normcdflog` | [ ] | [ ] | [ ] | Log normal CDF |
| 175 | `v_besselratio` | [ ] | [ ] | [ ] | Bessel function ratio |
| 176 | `v_besselratioi` | [ ] | [ ] | [ ] | Inverse Bessel ratio |
| 177 | `v_besratinv0` | [ ] | [ ] | [ ] | Bessel ratio inverse |

---

## Phase 8: Speech Analysis & Enhancement (0 / 28)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 178 | `v_melbankm` | [ ] | [ ] | [ ] | Mel filterbank matrix |
| 179 | `v_melcepst` | [ ] | [ ] | [ ] | MFCC extraction |
| 180 | `v_cep2pow` | [ ] | [ ] | [ ] | Cepstral → power |
| 181 | `v_pow2cep` | [ ] | [ ] | [ ] | Power → cepstral |
| 182 | `v_ldatrace` | [ ] | [ ] | [ ] | LDA trace |
| 183 | `v_correlogram` | [ ] | [ ] | [ ] | Correlogram |
| 184 | `v_spgrambw` | [ ] | [ ] | [ ] | Spectrogram |
| 185 | `v_modspect` | [ ] | [ ] | [ ] | Modulation spectrum |
| 186 | `v_estnoiseg` | [ ] | [ ] | [ ] | Noise estimation (Gerkmann) |
| 187 | `v_estnoisem` | [ ] | [ ] | [ ] | Noise estimation (Martin) |
| 188 | `v_specsub` | [ ] | [ ] | [ ] | Spectral subtraction |
| 189 | `v_specsubm` | [ ] | [ ] | [ ] | Spectral subtraction (multi) |
| 190 | `v_ssubmmse` | [ ] | [ ] | [ ] | MMSE enhancement |
| 191 | `v_ssubmmsev` | [ ] | [ ] | [ ] | MMSE-V enhancement |
| 192 | `v_spendred` | [ ] | [ ] | [ ] | Enhancement + dereverb |
| 193 | `v_dypsa` | [ ] | [ ] | [ ] | Glottal closure detection |
| 194 | `v_fxpefac` | [ ] | [ ] | [ ] | Pitch extraction (PEFAC) |
| 195 | `v_fxrapt` | [ ] | [ ] | [ ] | Pitch extraction (RAPT) |
| 196 | `v_glotros` | [ ] | [ ] | [ ] | Rosenberg glottal model |
| 197 | `v_glotlf` | [ ] | [ ] | [ ] | Liljencrants-Fant glottal |
| 198 | `v_sapisynth` | [ ] | [ ] | [ ] | TTS synthesis |
| 199 | `v_importsii` | [ ] | [ ] | [ ] | Speech Intelligibility Index |
| 200 | `v_vadsohn` | [ ] | [ ] | [ ] | Voice activity detection |
| 201 | `v_phon2sone` | [ ] | [ ] | [ ] | Phon → sone |
| 202 | `v_sone2phon` | [ ] | [ ] | [ ] | Sone → phon |
| 203 | `v_mos2pesq` | [ ] | [ ] | [ ] | MOS → PESQ |
| 204 | `v_pesq2mos` | [ ] | [ ] | [ ] | PESQ → MOS |
| 205 | `v_stoi2prob` | [ ] | [ ] | [ ] | STOI → probability |

---

## Phase 9: Rotation, Quaternion & Geometry (0 / 30)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 206 | `v_roteu2qr` | [ ] | [ ] | [ ] | Euler → quaternion |
| 207 | `v_rotqr2eu` | [ ] | [ ] | [ ] | Quaternion → Euler |
| 208 | `v_rotro2qr` | [ ] | [ ] | [ ] | Rotation matrix → quaternion |
| 209 | `v_rotqr2ro` | [ ] | [ ] | [ ] | Quaternion → rotation matrix |
| 210 | `v_roteu2ro` | [ ] | [ ] | [ ] | Euler → rotation matrix |
| 211 | `v_rotro2eu` | [ ] | [ ] | [ ] | Rotation matrix → Euler |
| 212 | `v_rotmc2qc` | [ ] | [ ] | [ ] | Matrix col → quaternion col |
| 213 | `v_rotqc2mc` | [ ] | [ ] | [ ] | Quaternion col → matrix col |
| 214 | `v_rotqc2qr` | [ ] | [ ] | [ ] | Quaternion col → quat row |
| 215 | `v_rotqr2qc` | [ ] | [ ] | [ ] | Quaternion row → quat col |
| 216 | `v_rotmr2qr` | [ ] | [ ] | [ ] | Matrix row → quaternion |
| 217 | `v_rotqr2mr` | [ ] | [ ] | [ ] | Quaternion → matrix row |
| 218 | `v_rotax2qr` | [ ] | [ ] | [ ] | Axis-angle → quaternion |
| 219 | `v_rotqr2ax` | [ ] | [ ] | [ ] | Quaternion → axis-angle |
| 220 | `v_rotqrmean` | [ ] | [ ] | [ ] | Average quaternions |
| 221 | `v_rotqrvec` | [ ] | [ ] | [ ] | Rotate vectors by quaternion |
| 222 | `v_roteucode` | [ ] | [ ] | [ ] | Euler angle code |
| 223 | `v_rotation` | [ ] | [ ] | [ ] | General rotation utility |
| 224 | `v_rotpl2ro` | [ ] | [ ] | [ ] | Plane → rotation matrix |
| 225 | `v_rotro2pl` | [ ] | [ ] | [ ] | Rotation matrix → plane |
| 226 | `v_rotlu2ro` | [ ] | [ ] | [ ] | Look-up → rotation matrix |
| 227 | `v_rotro2lu` | [ ] | [ ] | [ ] | Rotation matrix → look-up |
| 228 | `v_imagehomog` | [ ] | [ ] | [ ] | Image homography |
| 229 | `v_rectifyhomog` | [ ] | [ ] | [ ] | Rectify homography |
| 230 | `v_polygonarea` | [ ] | [ ] | [ ] | Polygon area |
| 231 | `v_polygonwind` | [ ] | [ ] | [ ] | Polygon winding number |
| 232 | `v_polygonxline` | [ ] | [ ] | [ ] | Polygon × line intersection |
| 233 | `v_minspane` | [ ] | [ ] | [ ] | Minimum spanning tree |
| 234 | `v_upolyhedron` | [ ] | [ ] | [ ] | Uniform polyhedron |
| 235 | `v_skew3d` | [ ] | [ ] | [ ] | 3D skew-symmetric matrix |

---

## Phase 10: Plotting, Display & Misc (0 / 47)

### Plotting & Display (0 / 20)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 236 | `v_axisenlarge` | [ ] | [ ] | [ ] | Axis enlargement (23 callers) |
| 237 | `v_texthvc` | [ ] | [ ] | [ ] | Text placement (19 callers) |
| 238 | `v_figbolden` | [ ] | [ ] | [ ] | Bold figure elements |
| 239 | `v_fig2pdf` | [ ] | [ ] | [ ] | Figure → PDF |
| 240 | `v_fig2emf` | [ ] | [ ] | [ ] | Figure → EMF |
| 241 | `v_cblabel` | [ ] | [ ] | [ ] | Colorbar label |
| 242 | `v_xticksi` | [ ] | [ ] | [ ] | X-axis SI ticks |
| 243 | `v_yticksi` | [ ] | [ ] | [ ] | Y-axis SI ticks |
| 244 | `v_xyzticksi` | [ ] | [ ] | [ ] | XYZ-axis SI ticks |
| 245 | `v_xtickint` | [ ] | [ ] | [ ] | X-axis integer ticks |
| 246 | `v_ytickint` | [ ] | [ ] | [ ] | Y-axis integer ticks |
| 247 | `v_tilefigs` | [ ] | [ ] | [ ] | Tile figure windows |
| 248 | `v_colormap` | [ ] | [ ] | [ ] | Custom colormaps |
| 249 | `v_lambda2rgb` | [ ] | [ ] | [ ] | Wavelength → RGB |
| 250 | `v_sprintcpx` | [ ] | [ ] | [ ] | Sprint complex number |
| 251 | `v_sprintsi` | [ ] | [ ] | [ ] | Sprint SI-prefixed |
| 252 | `v_peak2dquad` | [ ] | [ ] | [ ] | 2D quadratic peak |
| 253 | `v_quadpeak` | [ ] | [ ] | [ ] | Quadratic peak interp |
| 254 | `v_modsym` | [ ] | [ ] | [ ] | Modular symmetry |
| 255 | `v_sigma` | [ ] | [ ] | [ ] | Sigma function |

### Math & Linear Algebra (0 / 10)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 256 | `v_atan2sc` | [ ] | [ ] | [ ] | atan2 sin/cos |
| 257 | `v_dlyapsq` | [ ] | [ ] | [ ] | Discrete Lyapunov |
| 258 | `v_dualdiag` | [ ] | [ ] | [ ] | Dual diagonalization |
| 259 | `v_hypergeom1f1` | [ ] | [ ] | [ ] | Hypergeometric 1F1 |
| 260 | `v_mintrace` | [ ] | [ ] | [ ] | Minimum trace |
| 261 | `v_qrabs` | [ ] | [ ] | [ ] | QR absolute |
| 262 | `v_qrdivide` | [ ] | [ ] | [ ] | QR divide |
| 263 | `v_qrdotdiv` | [ ] | [ ] | [ ] | QR dot divide |
| 264 | `v_qrdotmult` | [ ] | [ ] | [ ] | QR dot multiply |
| 265 | `v_qrmult` | [ ] | [ ] | [ ] | QR multiply |
| 266 | `v_qrpermute` | [ ] | [ ] | [ ] | QR permute |

### Distance Measures (0 / 7)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 267 | `v_disteusq` | [ ] | [ ] | [ ] | Euclidean squared dist |
| 268 | `v_distchar` | [ ] | [ ] | [ ] | COSH dist (AR coeff) |
| 269 | `v_distchpf` | [ ] | [ ] | [ ] | COSH dist (power spec) |
| 270 | `v_distisar` | [ ] | [ ] | [ ] | Itakura-Saito (AR) |
| 271 | `v_distispf` | [ ] | [ ] | [ ] | Itakura-Saito (power) |
| 272 | `v_distitar` | [ ] | [ ] | [ ] | Itakura dist (AR) |
| 273 | `v_distitpf` | [ ] | [ ] | [ ] | Itakura dist (power) |

### System & Misc (0 / 9)

| # | Function | Octave Ref | Python | Tests | Notes |
|---|----------|:----------:|:------:|:-----:|-------|
| 274 | `v_hostipinfo` | [ ] | [ ] | [ ] | Host/IP information |
| 275 | `v_unixwhich` | [ ] | [ ] | [ ] | Unix which |
| 276 | `v_regexfiles` | [ ] | [ ] | [ ] | Regex file matching |
| 277 | `v_winenvar` | [ ] | [ ] | [ ] | Windows env variables |
| 278 | `v_voicebox_update` | [ ] | [ ] | [ ] | Update function prefixes |
| 279 | `v_m2htmlpwd` | [ ] | [ ] | [ ] | HTML doc generator |
| 280 | `v_psycdigit` | [ ] | [ ] | [ ] | Psychoacoustic digit |
| 281 | `v_psycest` | [ ] | [ ] | [ ] | Psychoacoustic estimation |
| 282 | `v_psycestu` | [ ] | [ ] | [ ] | Psychoacoustic est. (u) |
| 283 | `v_psychofunc` | [ ] | [ ] | [ ] | Psychometric function |
| 284 | `v_soundspeed` | [ ] | [ ] | [ ] | Speed of sound |

---

## Legacy Wrappers (0 / 258)

These are thin wrappers without the `v_` prefix. They will be auto-generated as Python re-exports once the core functions are converted.

**Status**: Not started — will be batch-generated after all core functions are complete.

---

## Constants & Demos (0 / 4)

| # | File | Status | Notes |
|---|------|:------:|-------|
| 1 | `v_rotro2eu_tab` | [ ] | Rotation lookup table → Python dict |
| 2 | `v_estnoiseg_d` | [ ] | Noise estimation demo |
| 3 | `v_readwav_d` | [ ] | Audio I/O demo |
| 4 | `v_windows_d` | [ ] | Window function demo |

---

## Legend

- `[ ]` = Not started
- `[~]` = In progress
- `[x]` = Complete
- `[-]` = Skipped (not applicable in Python)
