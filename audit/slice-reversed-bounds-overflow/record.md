# Reversed slice bounds cause an arithmetic-overflow panic

- **Component:** core crate — `safetensors/src/slice.rs` (`slice_byte_ranges`)
- **Reachable from:** Python `safe_open(...).get_slice(name)[hi:lo]`; Rust `TensorView::slice(10..3)`
- **Class:** integer-overflow panic → Denial of Service / unexpected crash
- **Severity:** Low (no memory unsafety; Rust bounds-checks the eventual slice). DoS only.
- **Found by:** `fuzz_slice` cargo-fuzz target (`safetensors/fuzz/fuzz_targets/fuzz_slice.rs`)

## Summary

When a `Narrow` slice resolves to `start > stop` (with both indices within the
dimension), `slice_byte_ranges` computes `stop - start` on `usize` and
underflows:

```
slice.rs:377   newshape.push((stop - start).div_ceil(step));
slice.rs:390   let small_span = (stop * span) / 8 - offset;   // same underflow, contiguous path
```

The guard immediately above only checks the upper edge and never rejects
reversed bounds:

```
slice.rs:363   if start >= dim || stop > dim { return Err(SliceOutOfRange { .. }); }
```

Python's `parse_indexers` maps `slice(hi, lo)` to
`Narrow(Included(hi), Excluded(lo))`, so any ordinary reversed slice reaches
`start > stop`. Every other Python slicing API (list, numpy, torch) treats a
reversed slice as a **benign empty selection**; safetensors instead panics.

## Behaviour by build profile

- **Debug / `overflow-checks` on** (incl. the default `maturin develop` build):
  immediate panic `attempt to subtract with overflow` at `slice.rs:377`,
  surfaced through PyO3 as `pyo3_runtime.PanicException`.
- **Release wheels (`overflow-checks` off):** the subtraction wraps to a huge
  `usize`, producing an inverted byte range `(40, 12)`; `SliceIterator::next`
  then does `data()[40..12]` → **panic `slice index starts at 40 but ends at
  12` at `slice.rs:440`** (an out-of-bounds slice, which is bounds-checked in
  release too). Via the numpy binding the wrapped length hits
  `PyByteArray_FromStringAndSize` first and surfaces as
  **`SystemError: Negative size passed to PyByteArray_FromStringAndSize`**.
  **Verified in release** (both the Rust API and a `maturin develop --release`
  binding) — this is the slicing case that triggers in release, and is packaged
  as a HuggingFace submission (`slice_panic.safetensors`, `repro.py`,
  `REPORT.md`).

## Trigger / test files

- `trigger.py` — self-contained Python reproducer (verified against the built
  binding). `f.get_slice("w")[10:3]` raises `PanicException`, while the numpy
  equivalent `arr[10:3]` returns an empty array.
- `trigger_rust.rs` — the same bug from the Rust API via `view.slice(10..3)`.
  Build as a standalone crate (see the header comment in the file).
- `slice_panic.safetensors` — a valid PoC model file (loads normally) for the
  HuggingFace submission; sliced with `[10:3]` it triggers the release panic.
- `repro.py` — loads `slice_panic.safetensors` and slices `[10:3]`; prints the
  release-mode `SystemError`. See `REPORT.md` for the submission writeup + HF
  upload steps.
- `../../safetensors/fuzz/fuzz_targets/fuzz_slice.rs` — the fuzz target that
  discovered it (structured input:
  `dtype=C64, dims=[255,33], indexers=[Narrow(Excluded(_), Excluded(0), _)]`,
  reducing to `start > stop`).

## Observed output (Python)

```
numpy [10:3] -> shape (0, 4) (no error)
thread '<unnamed>' panicked at src/slice.rs:377:31: attempt to subtract with overflow
safetensors [10:3] -> RAISED pyo3_runtime.PanicException
```

## Observed output (Rust standalone, debug)

```
thread 'main' panicked at src/slice.rs:377:31: attempt to subtract with overflow
```

## Suggested fix (not applied)

In `slice_byte_ranges`, after resolving `(start, stop)`, treat `start >= stop`
as an empty selection (push a `0` extent to `newshape` and an empty/zero-length
byte range), matching Python slice semantics; keep the existing
`SliceOutOfRange` error for `start > dim`. Use checked/saturating arithmetic
for the span computations.
