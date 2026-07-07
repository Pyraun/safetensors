# `fuzz_slice` — slicing-path fuzz target

Stresses the slice arithmetic in `src/slice.rs` (`slice_byte_ranges` /
`SliceIterator`) that a caller reaches through `TensorView::sliced_data`
and, in the Python bindings, through `safe_open(...).get_slice(name)[...]`.

Each case builds a *validatable* `TensorView` (via `TensorView::new`, so the
shape is representable and byte-aligned) and applies an arbitrary sequence of
`TensorIndexer`s. Because the tensor itself is always valid, any panic comes
from the indexer arithmetic. cargo-fuzz builds with `overflow-checks` on, so a
wrapping add/subtract/multiply aborts and is reported.

## Run

```
cd safetensors
cargo +nightly fuzz run fuzz_slice -- -max_total_time=60
```

## Findings (as of this target's introduction)

Both are reachable arithmetic-overflow panics in `narrow_bounds` /
`slice_byte_ranges`. The bounds/range check at `slice.rs:363`
(`start >= dim || stop > dim`) runs *after* the offending arithmetic and does
not enforce `start <= stop`.

### 1. Reversed-bounds subtract-overflow — reachable from Python

`slice.rs:377` `newshape.push((stop - start).div_ceil(step))` (and the
analogous `(stop * span) / 8 - offset` at `slice.rs:390`) underflow when a
`Narrow` slice resolves to `start > stop` with both within the dimension.

Python's `parse_indexers` maps `slice(hi, lo)` to
`Narrow(Included(hi), Excluded(lo))`, so this triggers on an ordinary reversed
slice that every other Python slicing API treats as a benign empty result:

```python
import numpy as np
from safetensors.numpy import save_file
from safetensors import safe_open

save_file({"w": np.zeros((16, 4), dtype=np.float32)}, "m.safetensors")
with safe_open("m.safetensors", framework="np") as f:
    np.zeros((16, 4))[10:3]      # numpy: empty (0, 4), no error
    f.get_slice("w")[10:3]       # safetensors: PanicException / crash
```

- Debug / `overflow-checks` builds (incl. default `maturin develop`):
  `pyo3_runtime.PanicException: attempt to subtract with overflow`.
- Release wheels (no `overflow-checks`): the subtraction wraps to a huge
  `usize`, which flows into the byte-range math and finally an out-of-bounds
  `data()[start..stop]` in `SliceIterator::next` — still a panic/exception.

Impact: DoS / unexpected crash in any code that slices with computed bounds
where `start` can exceed `stop` (windowing, negative-index math, etc.).

### 2. Bound-at-`usize::MAX` add-overflow — Rust API only

`narrow_bounds` (`slice.rs:320` `Excluded(s) => *s + 1`, `slice.rs:324`
`Included(s) => *s + 1`) overflows when the bound is `usize::MAX`. Not
reachable from the Python bindings (which only emit `Included` start /
`Excluded` stop bounds), but reachable from the Rust API via an inclusive
range ending at `usize::MAX`, e.g. `view.slice(..=usize::MAX)`.

## Suggested fix direction (not applied here)

In `slice_byte_ranges`, resolve bounds with saturating/checked arithmetic and
clamp `stop = stop.max(start)` (or return an empty selection when
`start >= stop`, matching Python semantics) before the subtraction and
multiplications; keep the existing out-of-range error for `start > dim`.
