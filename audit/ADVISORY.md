# Security Advisory — Integer-overflow panics (DoS) in `safetensors`

> Draft for a GitHub Security Advisory on `huggingface/safetensors`. The fields
> below map to the "New draft security advisory" form; the prose beneath is the
> advisory body. Discovered via a `cargo-fuzz` target for the slicing path.

## Advisory metadata

| Field | Value |
|---|---|
| **Ecosystem / package** | crates.io — `safetensors` (Rust); also affects the PyPI `safetensors` Python bindings, which wrap the same crate |
| **Affected versions** | Current `main` (crate `0.9.0-dev.0`) and prior releases exposing `TensorView::slice` / `SafeTensors::read_metadata` — the affected code paths (`slice::slice_byte_ranges`, `tensor::read_metadata`) are long-standing; maintainers to confirm the exact released range |
| **Patched versions** | _to be assigned_ |
| **Severity** | Low (Denial of Service; no memory corruption, no code execution, no data disclosure) |
| **CVSS v3.1 (estimate)** | `AV:L/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L` = **3.3 Low** (raise `AV:N` → 4.0 where model files / slice bounds arrive over a network boundary) |
| **CWE** | CWE-190 (Integer Overflow), CWE-191 (Integer Underflow), CWE-617 (Reachable panic) → CWE-400 (Uncontrolled Resource Consumption / DoS) |

## Summary

Two independent integer-overflow bugs in the `safetensors` Rust core cause the
library to **panic** (Rust) / raise an unexpected fatal error (Python), rather
than returning an error, on certain inputs:

1. **Reversed slice bounds** (`start > stop`) in `slice::slice_byte_ranges` —
   reachable from `safe_open(...).get_slice(name)[hi:lo]` (Python) and
   `TensorView::slice` (Rust). **Reproduces in release builds.** (A sibling
   `usize::MAX`-index overflow in the same function is already handled by pending
   PR #809 and is excluded here.)
2. **Header buffer-length check** in `tensor::read_metadata` — reachable from
   `deserialize()` / `load()` on a crafted file whose tensor offsets sum to
   `usize::MAX`. **Panics only in `overflow-checks` builds**; release wheels
   reject the file gracefully (verified — no bypass).

Both are Denial-of-Service only. There is no out-of-bounds read/write beyond a
Rust-checked slice panic, no code execution, and (for bug 2) no validation
bypass.

---

## Bug 1 — Reversed slice bounds panic (`src/slice.rs`)

### Description

`slice_byte_ranges` resolves each dimension's `(start, stop)` in `narrow_bounds`
and range-checks it at `src/slice.rs:363`:

```rust
if start >= dim || stop > dim { return Err(InvalidSlice::SliceOutOfRange { .. }); }
```

This check never rejects `start > stop`. A reversed slice (e.g. `[10:3]`) then
underflows `stop - start` at `src/slice.rs:377`:

```rust
newshape.push((stop - start).div_ceil(step));      // underflow when start > stop
// ... and the analogous (stop * span) / 8 - offset at src/slice.rs:390
```

> **Note — a sibling overflow is already being fixed.** `narrow_bounds`
> (`src/slice.rs:320`, `:324`) and `Select` (`:357`) also do `*s + 1` without
> overflow checking, so a `usize::MAX` index overflows. That specific sub-issue
> is addressed by pending upstream PR
> [#809](https://github.com/safetensors/safetensors/pull/809) ("Guard slice
> bound resolution against index overflow", `saturating_add`). **This advisory
> excludes it** and covers only the reversed-bounds (`start > stop`) case, which
> PR #809 does **not** fix — its `saturating_add` leaves the `stop - start`
> underflow at `:377` intact, so the panic below still reproduces with #809
> applied.

### Reachability & build-profile behavior

| Input | Debug / `overflow-checks` | **Release (published wheels)** |
|---|---|---|
| `get_slice(name)[10:3]` (reversed) | panic `attempt to subtract with overflow` @ `slice.rs:377` | subtraction wraps → inverted byte range `(40, 12)` → `SliceIterator::next` does `data()[40..12]` → **panic `slice index starts at 40 but ends at 12` @ `slice.rs:440`**. Via the numpy binding: **`SystemError: Negative size passed to PyByteArray_FromStringAndSize`**. |

Reachable from the numpy/tf/flax/mlx byte-slice path and the Rust crate API;
also the torch/paddle path under `backend="pread"`. Every other Python slicing
API (`list`, numpy, torch) treats a reversed slice as a benign empty selection.

Note: the slice bounds come from the **caller**, not the file, so a plain
`load_file` does not trigger this. Impact lands on code that slices with
computed bounds where `start` can exceed `stop` (windowing, sharding,
negative-index arithmetic).

### Proof of concept

```python
import numpy as np
from safetensors.numpy import save_file
from safetensors import safe_open

save_file({"weight": np.zeros((16, 16), dtype=np.float32)}, "m.safetensors")
np.zeros((16, 16))[10:3]                        # numpy: (0, 16), no error
with safe_open("m.safetensors", framework="np") as f:
    f.get_slice("weight")[10:3]                 # safetensors: crashes (release too)
```

Rust: `TensorView::new(Dtype::F32, vec![16], &data).unwrap().slice(10..3)`
panics `slice index starts at 40 but ends at 12` in `--release`.

### Suggested fix

In `slice_byte_ranges`, treat `start >= stop` as an empty selection (push a `0`
extent to `newshape` and an empty byte range), matching Python slice semantics,
and use checked/saturating arithmetic for the `start*span` / `stop*span`
computations. Keep the existing `SliceOutOfRange` error for `start > dim`.
(PR #809's `saturating_add` in `narrow_bounds`/`Select` is complementary but
does **not** cover `start > stop`, so this fix is still required on top of it.)

---

## Bug 2 — Header buffer-length check overflow (`src/tensor.rs`)

### Description

`SafeTensors::read_metadata` validates that the declared tensor data exactly
covers the buffer at `src/tensor.rs:420`:

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
20M-sample arithmetic Monte Carlo (`wrap-to-pass = 0`). So release wheels are
not affected; only `overflow-checks` builds panic.

### Proof of concept

```python
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
panic/fatal error. No memory corruption, code execution, or data disclosure.

## Workarounds

- **Bug 1:** callers should clamp slice bounds so `start <= stop` (and `stop <=
  dim`) before calling `get_slice(...)[...]`, treating reversed ranges as empty.
- **Bug 2:** build/ship in `--release` without `overflow-checks` (the default
  for published wheels) — release wheels already reject the crafted file
  gracefully. To also harden `overflow-checks` builds, apply the fix above.

## Detection / discovery

Found by a `cargo-fuzz` target (`fuzz_slice`) that crosses a validated
`TensorView` with arbitrary `TensorIndexer` sequences; both slice panics were
found within seconds. The buffer-length overflow was found while verifying the
`read_metadata` coverage check.

## Related / prior art

- PR [#809](https://github.com/safetensors/safetensors/pull/809) — "Guard slice
  bound resolution against index overflow" (open, not merged as of upstream
  `main` @ `6eb4dc9`). Fixes the sibling `usize::MAX`-index `*s + 1` overflow via
  `saturating_add`; does **not** address Bug 1's reversed-bounds case or Bug 2.

## References

- `src/slice.rs` — `narrow_bounds` (`:316`), `slice_byte_ranges` (`:363`, `:377`,
  `:390`), `SliceIterator::next` (`:440`)
- `src/tensor.rs` — `read_metadata` (`:420`)
- PoCs and verification harnesses: `audit/slice-reversed-bounds-overflow/`,
  `audit/buffer-length-overflow/`, and the `fuzz_slice` target under
  `safetensors/fuzz/`.
