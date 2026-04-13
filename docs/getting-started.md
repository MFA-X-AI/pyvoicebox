---
icon: material/rocket-launch
---

# Getting Started

## Installation

```bash
pip install py-voicebox                # core (numpy, scipy, soundfile)
pip install "py-voicebox[plot]"        # with matplotlib for plotting functions
```

For development (includes pytest and matplotlib):

```bash
pip install -e ".[dev]"
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations and linear algebra |
| `scipy` | Signal processing, special functions, sparse matrices, optimization |
| `soundfile` | Audio file I/O (WAV, FLAC, AIFF, AU) via libsndfile |
| `matplotlib` | *(optional)* Plotting and display functions |

## Quick Start

### Frequency conversions

```python
from pyvoicebox import v_frq2mel, v_mel2frq
mel = v_frq2mel(440)          # Hz to Mel
hz = v_mel2frq(mel)           # back to Hz
```

### MFCC extraction

```python
from pyvoicebox import v_melcepst
import soundfile as sf
signal, fs = sf.read('speech.wav')
mfcc = v_melcepst(signal, fs, 'M0dD', 12)  # 12 MFCCs + deltas
```

### LPC analysis

```python
from pyvoicebox import v_lpcauto, v_lpcar2cc
ar, e, k = v_lpcauto(signal, 12)     # 12th-order LPC
cc = v_lpcar2cc(ar)                    # AR -> cepstral coefficients
```

### Quaternion operations

```python
from pyvoicebox import v_roteu2qr, v_rotqr2ro
import numpy as np
q = v_roteu2qr('xyz', np.array([0.1, 0.2, 0.3]))  # Euler -> quaternion
R = v_rotqr2ro(q)                                     # quaternion -> rotation matrix
```

### Noise estimation & speech enhancement

```python
from pyvoicebox import v_estnoiseg, v_specsub
noise_psd = v_estnoiseg(signal, fs)       # noise PSD estimate
clean = v_specsub(signal, fs)             # spectral subtraction
```

## Function naming

All functions are available with both the `v_` prefix (matching the MATLAB original) and without:

```python
from pyvoicebox import frq2mel       # same as v_frq2mel
from pyvoicebox import melcepst      # same as v_melcepst
from pyvoicebox import lpcauto       # same as v_lpcauto
```

The `v_` prefix avoids naming collisions and makes it easy to grep for VOICEBOX functions in your codebase. Use whichever style you prefer.

## Type annotations

All functions have return type annotations, so your IDE can show what each function returns:

```python
v_frq2mel(frq) -> tuple[np.ndarray, np.ndarray]
v_lpcauto(s, p=12, ...) -> tuple[np.ndarray, np.ndarray, np.ndarray]
v_midi2frq(n, s='e') -> np.ndarray
v_writewav(d, fs, filename, ...) -> None
```

Many VOICEBOX functions return tuples — for example, `v_frq2mel` returns `(mel_values, gradient)`. The type annotations make this visible without reading the docstring.

## Next steps

- Browse the [API Reference](api/audio-io.md) to find functions by category
- See how pyvoicebox compares to [librosa and openSMILE](comparison.md)
- Read about the [testing approach](testing.md) and how correctness is verified
