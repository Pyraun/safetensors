# Security Advisory — Integer-overflow panics (Denial of Service) in `safetensors`

**Reporter:** _<your name / contact>_
**Date:** _<fill in>_
**Repository:** `safetensors/safetensors` (`huggingface/safetensors` redirects here)
**Status:** private disclosure to the maintainers' security team

This is a self-contained report; every proof of concept below is inlined and
runnable without any additional files. It also maps cleanly onto a GitHub
"New draft security advisory" (the metadata table = the form fields; the prose
= the advisory body).

## Advisory metadata

| Field | Value |
|---|---|
| **Ecosystem / package** | crates.io — `safetensors` (Rust); also affects the PyPI `safetensors` Python bindings, which wrap the same crate |
| **Affected versions** | Current `main` (crate `0.9.0-dev.0`) and prior releases exposing `TensorView::slice` / `SafeTensors::read_metadata`. The affected code paths (`slice::slice_byte_ranges`, `tensor::read_metadata`) are long-standing; maintainers to confirm the exact released range. |
| **Patched versions** | _to be assigned_ |
| **Severity** | Low (Denial of Service; no memory corruption, no code execution, no data disclosure) |
| **CVSS v3.1 (estimate)** | `AV:L/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L` = **3.3 Low** (raise `AV:N` → 4.0 where model files / slice bounds arrive over a network boundary) |
| **CWE** | CWE-190 (Integer Overflow), CWE-191 (Integer Underflow), CWE-617 (Reachable panic) → CWE-400 (Uncontrolled Resource Consumption / DoS) |

## Summary

Two independent integer-overflow bugs in the `safetensors` Rust core cause the
library to **panic** (Rust) / raise an unexpected fatal error (Python), rather
than returning an `Err`, on certain inputs:

1. **Reversed slice bounds** (`start > stop`) in `slice::slice_byte_ranges` —
   reachable from `safe_open(...).get_slice(name)[hi:lo]` (Python) and
   `TensorView::slice` (Rust). **Reproduces in release builds** (published
   wheels).
2. **Header buffer-length check** in `tensor::read_metadata` — reachable from
   `deserialize()` / `load()` on a crafted file whose tensor offsets sum to
   `usize::MAX`. **Panics only in `overflow-checks` builds**; release wheels
   reject the file gracefully (verified — no bypass).

Both are Denial-of-Service only. There is no out-of-bounds read/write beyond a
Rust-checked slice panic, no code execution, and (for bug 2) no validation
bypass.

A related, in-flight PR (#809) fixes a *sibling* overflow in the same slicing
function but does **not** address either bug here; see "Relationship to PR #809"
below, where we propose the additional one-line fix that closes Bug 1.

---

## Bug 1 — Reversed slice bounds panic (`src/slice.rs`)

### Description

`slice_byte_ranges` resolves each dimension's `(start, stop)` and range-checks
it (in current `main`, around `src/slice.rs:363`):

```rust
if start >= dim || stop > dim { return Err(InvalidSlice::SliceOutOfRange { .. }); }
```

This check never rejects `start > stop`. A reversed slice (e.g. `[10:3]`) then
underflows `stop - start` at `src/slice.rs:377`:

```rust
newshape.push((stop - start).div_ceil(step));      // underflow when start > stop
// ... and the analogous (stop * span) / 8 - offset at src/slice.rs:390
```

### Reachability & build-profile behavior

| Input | Debug / `overflow-checks` | **Release (published wheels)** |
|---|---|---|
| `get_slice(name)[10:3]` (reversed) | panic `attempt to subtract with overflow` @ `slice.rs:377` | subtraction wraps → inverted byte range `(40, 12)` → `SliceIterator::next` does `data()[40..12]` → **panic `slice index starts at 40 but ends at 12` @ `slice.rs:440`**. Via the numpy binding this surfaces as **`SystemError: Negative size passed to PyByteArray_FromStringAndSize`** (the wrapped length becomes a negative `ssize_t`). |

Reachable from the numpy / tensorflow / flax / mlx byte-slice path and the Rust
crate API; also the torch / paddle path under `backend="pread"`. Every other
Python slicing API (`list`, numpy, torch) treats a reversed slice as a benign
empty selection.

Scope note: the slice bounds come from the **caller**, not the file, so a plain
`load_file` does not trigger this. Impact lands on code that slices with
computed bounds where `start` can exceed `stop` (windowing, sharding,
negative-index arithmetic).

### Proof of concept (self-contained)

```python
# pip install safetensors numpy      (a stock release wheel is enough)
import numpy as np
from safetensors.numpy import save_file
from safetensors import safe_open

save_file({"weight": np.zeros((16, 16), dtype=np.float32)}, "m.safetensors")
print(np.zeros((16, 16))[10:3].shape)              # numpy: (0, 16) — benign empty, no error
with safe_open("m.safetensors", framework="np") as f:
    f.get_slice("weight")[10:3]                    # safetensors: crashes (release too)
```

Expected (release wheel): the process raises
`SystemError: Negative size passed to PyByteArray_FromStringAndSize` and exits
non-zero.

Rust equivalent (panics `slice index starts at 40 but ends at 12` under
`--release`):

```rust
use safetensors::slice::IndexOp;
use safetensors::tensor::{Dtype, TensorView};
let data = vec![0u8; 16 * 4];                       // 16 x F32
let view = TensorView::new(Dtype::F32, vec![16], &data).unwrap();
let _ = view.slice(10..3).map(|it| it.count());     // 10..3 -> start=10, stop=3
```

### Suggested fix

Treat `start >= stop` as an empty selection (matching Python slice semantics)
before the subtraction. The minimal, robust fix is to clamp `stop` up to
`start` immediately after the bounds are resolved, so a reversed range becomes a
zero-length selection instead of underflowing:

```rust
            let (start, stop, step) = match slice {
                TensorIndexer::Select(s) => (*s, s.saturating_add(1), 1),
                TensorIndexer::Narrow(left, right, step) => {
                    let (start, stop) = narrow_bounds(left, right, dim);
                    (start, stop, step.get())
                }
            };
+           // A reversed range (start > stop) is an empty selection in every
+           // other slicing API (list/numpy/torch). Clamp so it yields an empty
+           // result instead of underflowing `stop - start` / `stop*span` below.
+           let stop = stop.max(start);
            if start >= dim || stop > dim {
                // existing SliceOutOfRange handling
            }
```

With this clamp, `newshape` gets a `0` extent for that dimension and the byte
range collapses to empty — the caller receives an empty tensor, exactly like
numpy/torch. The existing `SliceOutOfRange` error for `start > dim` is
preserved. (Verified: the clamp makes `[10:3]` return empty in both debug and
release.)

### Relationship to PR #809

Upstream PR **[#809](https://github.com/safetensors/safetensors/pull/809)**
("Guard slice bound resolution against index overflow", open, not merged as of
`main` @ `6eb4dc9`) fixes a *sibling* overflow — the `*s + 1` in `narrow_bounds`
/ `Select` when an index is `usize::MAX` — by switching to `saturating_add`.
That change is good and orthogonal, but it does **not** fix this bug: we built
the crate at PR #809's commit and confirmed `view.slice(10..3)` still panics in
release (`slice index starts at 40 but ends at 12`), because the `stop - start`
underflow is untouched. **We recommend the one-line `let stop = stop.max(start);`
clamp above be added to PR #809** (same function, same author's context): #809's
`saturating_add` already closes the `usize::MAX`-index case, and the clamp closes
the reversed-bounds case, so the amended PR fixes both in one place. We verified
the combination on PR #809's code — in both debug and release, `[10:3]` returns
an empty tensor, `..=usize::MAX` returns `SliceOutOfRange`, and normal slices are
unaffected.

---

## Bug 2 — Header buffer-length check overflow (`src/tensor.rs`)

### Description

`SafeTensors::read_metadata` validates that the declared tensor data exactly
covers the buffer (current `main`, `src/tensor.rs:420`):

```rust
if buffer_end + N_LEN + n != buffer_len {
    return Err(SafeTensorError::MetadataIncompleteBuffer);
}
```

`buffer_end` is the cumulative last tensor offset (up to `usize::MAX`). The
addition `buffer_end + N_LEN + n` is **unchecked**. A crafted file whose tensor
offsets accumulate to exactly `usize::MAX` (e.g. nine contiguous `U8` tensors)
overflows the `+ N_LEN`.

### Reachability & build-profile behavior

| Build | Behavior on the crafted file |
|---|---|
| Debug / `overflow-checks` (default `maturin develop`, `cargo test`, and any release consumer that sets `overflow-checks = true`) | **panic** `attempt to add with overflow` @ `tensor.rs:420` → `pyo3_runtime.PanicException` |
| Release, default (published PyPI wheels) | wraps silently, then `wrapped != buffer_len` → **graceful** `MetadataIncompleteBuffer` |

Reachable straight from `deserialize()` / `load()` (and `safe_open` /
`load_file`) via file content alone — this is a genuine model-file-parsing bug,
but its impact is limited to `overflow-checks` builds.

### No validation bypass (verified)

We checked whether an attacker could instead make the *wrapped* sum equal
`buffer_len` (passing the check, then reaching an out-of-bounds slice that would
panic even in release). This is **impossible**: a wrapping pass requires
`buffer_end >= 2^64`, but `buffer_end` is a `usize` capped at `2^64 - 1` while
the buffer must be at least `N_LEN + n` bytes. Confirmed with a real
`--release` deserialize sweep (`panics=0, bypass=0, closest gap = 1 byte`) and a
20-million-sample arithmetic Monte Carlo (`wrap-to-pass = 0`). So release wheels
are not affected; only `overflow-checks` builds panic.

### Proof of concept (self-contained)

```python
# Reproduces on an overflow-checks build (e.g. `maturin develop`, cargo test).
import json, struct
from safetensors import deserialize

MAXSZ = ((1 << 64) - 1) // 8            # 2305843009213693951 = largest valid U8 tensor size
meta, off = {}, 0
for i in range(8):                     # 8 * MAXSZ + 7 == 2**64 - 1
    meta[f"t{i}"] = {"dtype": "U8", "shape": [MAXSZ], "data_offsets": [off, off + MAXSZ]}
    off += MAXSZ
meta["t8"] = {"dtype": "U8", "shape": [7], "data_offsets": [off, off + 7]}   # cumulative == 2**64-1
body = json.dumps(meta, separators=(",", ":")).encode()
deserialize(struct.pack("<Q", len(body)) + body)   # overflow-checks build: PanicException
```

### Suggested fix

Use checked arithmetic at `tensor.rs:420`:

```rust
let expected = buffer_end.checked_add(N_LEN).and_then(|x| x.checked_add(n));
if expected != Some(buffer_len) {
    return Err(SafeTensorError::MetadataIncompleteBuffer);
}
```

This makes every build profile reject uniformly instead of panicking under
`overflow-checks`.

---

## Impact

Denial of Service. An application that (bug 1) slices safetensors tensors with
caller-computed bounds that can invert, or (bug 2) parses untrusted model files
in an `overflow-checks`-enabled build, can be crashed by an unexpected
panic / fatal error. No memory corruption, code execution, or data disclosure.

## Workarounds

- **Bug 1:** callers should clamp slice bounds so `start <= stop` (and
  `stop <= dim`) before calling `get_slice(...)[...]`, treating reversed ranges
  as empty.
- **Bug 2:** ship in `--release` without `overflow-checks` (the default for
  published wheels) — release wheels already reject the crafted file gracefully.
  To also harden `overflow-checks` builds, apply the fix above.

## Reproducing from scratch

```bash
# Python (Bug 1 on any release wheel; Bug 2 needs an overflow-checks build)
pip install safetensors numpy          # release wheel: Bug 1 crashes, Bug 2 rejects gracefully
# To also observe Bug 2's panic, build with overflow-checks on:
#   git clone https://github.com/safetensors/safetensors && cd safetensors/bindings/python
#   maturin develop                    # debug build -> overflow-checks on

# Rust core (both bugs, both profiles)
git clone https://github.com/safetensors/safetensors && cd safetensors/safetensors
cargo test        # debug: overflow-checks on
# build a small bin against the crate and run with --release to see release behavior
```

## Discovery

Bug 1 was found by a `cargo-fuzz` target that crosses a validated `TensorView`
with arbitrary `TensorIndexer` sequences (reversed and extreme bounds); it
surfaced within seconds. Bug 2 was found while auditing the `read_metadata`
buffer-coverage check for overflow.

## References (source locations, current `main`)

- `src/slice.rs` — `narrow_bounds` (`:316`), `slice_byte_ranges` (`:363`, `:377`,
  `:390`), `SliceIterator::next` (`:440`)
- `src/tensor.rs` — `read_metadata` (`:420`)
- PR #809 — https://github.com/safetensors/safetensors/pull/809 (sibling
  `usize::MAX`-index fix; see "Relationship to PR #809")
