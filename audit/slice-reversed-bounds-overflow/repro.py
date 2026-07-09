#!/usr/bin/env python3
"""Release-mode DoS via reversed slice bounds on a valid safetensors file.

The model file `slice_panic.safetensors` is a completely ordinary, valid
safetensors file (it loads fine). The bug is in the library's slicing API:
`safe_open(...).get_slice(name)[hi:lo]` with hi > lo — a reversed slice that
numpy/torch/list all treat as a benign empty selection — instead fails inside
the Rust slice arithmetic (src/slice.rs), and it fails *in release builds too*,
not only under debug/overflow-checks.

Reproduces on the published-wheel configuration (release):

    numpy binding : SystemError: Negative size passed to PyByteArray_FromStringAndSize
    rust API      : panic 'slice index starts at 40 but ends at 12' (src/slice.rs:440)

Run:
    pip install numpy safetensors
    python repro.py
"""
import numpy as np
from safetensors.numpy import load_file
from safetensors import safe_open

PATH = "slice_panic.safetensors"


def main() -> None:
    # The file is valid and loads normally.
    d = load_file(PATH)
    print("file loads fine:", {k: v.shape for k, v in d.items()})

    with safe_open(PATH, framework="np") as f:
        s = f.get_slice("weight")            # shape [16, 16]
        # numpy treats a reversed slice as an empty selection — no error:
        print("numpy  weight[10:3] ->", np.zeros((16, 16))[10:3].shape, "(benign empty)")
        # safetensors on the same reversed slice:
        try:
            out = s[10:3]
            print("safetensors weight[10:3] -> shape", out.shape, "(NO crash)")
        except BaseException as e:
            print("safetensors weight[10:3] -> RAISED",
                  type(e).__module__ + "." + type(e).__name__, ":",
                  str(e).strip().splitlines()[-1])


if __name__ == "__main__":
    main()
