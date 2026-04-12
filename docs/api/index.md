# API Reference

pyvoicebox provides **280+ functions** organised into 16 categories, following the same structure as the [original VOICEBOX documentation](https://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html).

## How to read this reference

Each function page is auto-generated from the Python source using [mkdocstrings](https://mkdocstrings.github.io/). You'll see:

- **Function signature** — parameters, types, and defaults
- **Docstring** — description, parameter details, return values, and examples
- **Source code** — click "Source code" to expand the full implementation

### Naming convention

Every function uses the `v_` prefix, matching the MATLAB original. Both prefixed and unprefixed names work:

```python
from pyvoicebox import v_frq2mel    # original name
from pyvoicebox import frq2mel      # also works
```

### MATLAB mode strings

Many functions accept a *mode string* — a compact string of flags that controls behaviour. For example:

```python
mfcc = v_melcepst(signal, fs, 'M0dD', 12)
#                              ^^^^
#  M = use mel scale, 0 = include C0, d = append deltas, D = append delta-deltas
```

Mode strings are documented in each function's docstring. They match the MATLAB originals exactly, so MATLAB documentation and papers remain valid references.

## Categories

| Category | Functions | Description |
|---|---|---|
| [Audio File I/O](audio-io.md) | 10 | Read and write WAV, HTK, SPHERE, AIFF, AU, FLAC, SFS, Connex |
| [Frequency Scale Conversion](frequency-scales.md) | 10 | Hz to/from Mel, Bark, ERB, Cent, MIDI |
| [Fourier, DCT & Hartley Transforms](transforms.md) | 9 | FFT, DCT, Hartley, zoom FFT, convolution |
| [Signal Processing](signal-processing.md) | 19 | Windows, filters, peaks, Teager energy, zero-crossings |
| [Speech Analysis](speech-analysis.md) | 35 | Framing, STFT, pitch (PEFAC, RAPT, DYPSA), VAD, psychoacoustics |
| [Speech Recognition & Features](speech-recognition.md) | 5 | MFCC, mel filterbanks, cepstrum/power conversion, LDA |
| [LPC Analysis](lpc.md) | 63 | Autocorrelation/covariance LPC + 54 representation conversions |
| [Speech Enhancement](speech-enhancement.md) | 7 | Noise estimation, spectral subtraction, MMSE, dereverberation |
| [Speech Synthesis](speech-synthesis.md) | 3 | SAPI TTS, Rosenberg and LF glottal flow models |
| [Speech Coding](speech-coding.md) | 5 | A-law, mu-law companding, POTS band filtering |
| [Random Numbers & Probability](probability.md) | 31 | RNGs, Gaussian mixtures, k-means, densities, Bessel functions |
| [Vector Distance](vector-distance.md) | 7 | Itakura, Itakura-Saito, COSH, Euclidean distance measures |
| [Information Theory](information-theory.md) | 2 | Huffman coding, entropy |
| [Rotations, Quaternions & Geometry](rotation-geometry.md) | 37 | Euler/matrix/quaternion conversions, polygon/polyhedron geometry |
| [Printing & Display](plotting.md) | 17 | Figure export, SI-prefix axes, colourmaps (requires `matplotlib`) |
| [Utility Functions](utility.md) | 22 | Configuration, filesystem, numeric helpers, combinatorics |
