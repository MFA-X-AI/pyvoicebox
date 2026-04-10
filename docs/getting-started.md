# Getting Started

## Installation

```bash
pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy`, `soundfile`, `matplotlib`. Tests additionally require `pytest`.

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

All functions are available with both the `v_` prefix (matching MATLAB) and without (legacy aliases):

```python
from pyvoicebox import frq2mel       # same as v_frq2mel
from pyvoicebox import melcepst      # same as v_melcepst
from pyvoicebox import lpcauto       # same as v_lpcauto
```

## Running tests

Tests validate against the original MATLAB source via GNU Octave:

```bash
pip install -e ".[dev]"
pytest tests/ -v
# First run: clones voicebox source, generates ref data via Octave, runs 511 tests
# Subsequent runs: uses cached ref data, runs tests only
```

### Prerequisites for running tests

- **Git** (to clone MATLAB source)
- **GNU Octave** (to generate reference data)
- **Python 3.9+** with numpy, scipy, soundfile, matplotlib
