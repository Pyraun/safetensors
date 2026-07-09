#!/usr/bin/env python3
"""Release-mode DoS via reversed slice bounds on a valid safetensors file.

The model file `slice_panic.safetensors` is a completely ordinary, valid
safetensors file (it loads fine). The bug is in the library's slicing API:
`safe_open(...).get_slice(name)[hi:lo]` with hi > lo — a reversed slice that
numpy/torch/list all treat as a benign empty selection — instead fails inside
the Rust slice arithmetic (src/slice.rs), and it fails *in release builds too*,
not only under debug/overflow-checks.

This script deliberately does NOT catch the error: it lets safetensors crash so
the impact is explicit (unhandled exception, full traceback, non-zero exit).

Expected final line:
    release wheel (overflow-checks off):
        SystemError: Negative size passed to PyByteArray_FromStringAndSize
    debug / overflow-checks build:
        pyo3_runtime.PanicException: attempt to subtract with overflow
    (at the Rust level, release panics 'slice index starts at 40 but ends at 12'
     at src/slice.rs:440.)

Run:
    pip install numpy safetensors        # release wheel
    python repro.py                      # exits non-zero with a traceback
"""
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file

PATH = "slice_panic.safetensors"

# The file is valid and loads normally.
d = load_file(PATH)
print("file loads fine:", {k: v.shape for k, v in d.items()})

# numpy treats a reversed slice as an empty selection — no error:
print("numpy  weight[10:3] ->", np.zeros((16, 16), dtype=np.float32)[10:3].shape,
      "(benign empty)")

# safetensors on the SAME reversed slice: let it crash (uncaught, by design).
print("safetensors weight[10:3] -> (expect crash below)")
with safe_open(PATH, framework="np") as f:
    f.get_slice("weight")[10:3]
