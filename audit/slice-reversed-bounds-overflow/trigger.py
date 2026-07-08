#!/usr/bin/env python3
"""Reversed-bounds slice overflow — Python reproducer.

A reversed slice (start > stop) is a benign empty selection everywhere in
Python (list/numpy/torch), but `safe_open(...).get_slice(name)[hi:lo]` panics
inside the Rust slicing arithmetic (src/slice.rs:377, `stop - start`).

Run:
    source /home/user/safetensors/.venv/bin/activate
    python trigger.py
"""

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

PATH = "reversed_slice.safetensors"


def main() -> None:
    save_file({"w": np.zeros((16, 4), dtype=np.float32)}, PATH)

    # Reference: numpy treats a reversed slice as empty, no error.
    ref = np.zeros((16, 4), dtype=np.float32)[10:3]
    print(f"numpy [10:3] -> shape {ref.shape} (no error)")

    with safe_open(PATH, framework="np") as f:
        s = f.get_slice("w")
        try:
            out = s[10:3]
            print(f"safetensors [10:3] -> shape {out.shape}  (UNEXPECTED: no panic)")
        except BaseException as e:  # PanicException is not a normal Exception subclass everywhere
            name = f"{type(e).__module__}.{type(e).__name__}"
            last = (str(e).strip().splitlines() or ["<empty>"])[-1]
            print(f"safetensors [10:3] -> RAISED {name}: {last}")


if __name__ == "__main__":
    main()
