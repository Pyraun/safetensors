# Buffer-length check: overflow panic (no logic bypass)

- **Component:** core crate — `SafeTensors::read_metadata` (`safetensors/src/tensor.rs:420`)
- **Reachable from:** `safetensors.deserialize()` / `load()` and `safe_open`/`load_file` (any path through `read_metadata`), purely via file content (tensor `data_offsets`). No slicing or user input needed.
- **Class:** integer-overflow panic → Denial of Service, **only in `overflow-checks` builds**
- **Severity:** Low. Published PyPI wheels (release, overflow-checks off) are **not** affected — they gracefully reject. Affects debug / `maturin develop` / `cargo test` builds and any downstream Rust consumer that enables `overflow-checks` in release.
- **Found by:** targeted probe while checking the logic-bypass hypothesis (`trigger.py`).

## Two questions, two answers

### 1. Can the check be *bypassed* (accept a truncated/oversized file)? — NO

The final validation is:

```
tensor.rs:420   if buffer_end + N_LEN + n != buffer_len {
                    return Err(SafeTensorError::MetadataIncompleteBuffer);
                }
```

`buffer_end` is the cumulative last offset returned by `validate()`. To wrap
`buffer_end + 8 + n` around `2^64` and land back on the real `buffer_len`, you
would need `buffer_end >= 2^64`. But offsets are `usize`, capped at `2^64 - 1`
(a JSON offset of `2^64` fails to parse — see case 3), and the file must be at
least `8 + n` bytes (header alone), which forces the non-wrapping solution.
Empirically, with `buffer_end` maxed at `2^64 - 1` the wrapped value is
`7 + n`, always `< 8 + n <= buffer_len`, so it can never equal `buffer_len`.
**No acceptance is possible.** Confirmed in release below.

### 2. Is the addition itself safe? — NO (panics on overflow)

`buffer_end + N_LEN + n` is **unchecked**. A crafted header whose offsets
accumulate to near `usize::MAX` (9 valid `U8` tensors summing to exactly
`2^64 - 1`) makes the `+ 8` overflow:

| Build profile | Behaviour |
|---|---|
| Debug / `overflow-checks = true` (default `maturin develop`, `cargo test`) | **panic** `attempt to add with overflow` at `tensor.rs:420` → `pyo3_runtime.PanicException` |
| Release, default (PyPI wheels, `overflow-checks` off) | wraps silently, then `wrapped != buffer_len` → **graceful** `MetadataIncompleteBuffer` |

The crafted file passes UTF-8, JSON, and `validate()` (offsets are internally
consistent and contiguous), so it reaches line 420 with `buffer_end = 2^64 - 1`.

## Observed output

Python (`trigger.py`, against the debug `maturin develop` binding):
```
[buffer_end == 2^64-1, 0 data bytes]
  need wrapped==buffer_len to bypass -> False
  ... panicked at src/tensor.rs:420:12: attempt to add with overflow
  RESULT: rejected -> PanicException: attempt to add with overflow
```

Rust standalone (`trigger_rust.rs`), both profiles:
```
=== DEBUG (overflow-checks on) ===
  panicked at src/tensor.rs:420:12: attempt to add with overflow
=== RELEASE (overflow-checks off) ===
  gracefully rejected: incomplete metadata, file not fully covered
```

## Release-mode verification (can the overflow be turned into a release panic?)

The debug panic is not the worst case. In **release** the add wraps silently,
so the real question is: can a crafted file make the *wrapped* sum **equal**
`buffer_len`, so the check **passes**, letting the astronomical offsets reach an
out-of-bounds slice in `deserialize`/`tensors()` — a panic that *does* fire in
release (slice indexing is always bounds-checked)?

**Answer: no. It is arithmetically impossible.** Writing `buffer_len = N_LEN + n + D`
(`D` = appended data bytes ≥ 0), the check `(buffer_end + N_LEN + n) mod 2^64 == buffer_len`
has only two solution families:

- **No wrap:** `buffer_end == D` — the *legitimate* fully-covered file. Every
  offset is then within the data section, so downstream slicing is in-bounds.
- **Wrap:** requires `buffer_end == 2^64 + D`, i.e. `buffer_end >= 2^64`. But
  `buffer_end` is a `usize` capped at `2^64 - 1` (offsets are `usize`; a JSON
  offset of `2^64` fails to parse). So a wrapping pass can never be constructed.

Equivalently: at the maximum `buffer_end = 2^64 - 1`, the wrapped value is
`7 + n`, while the file must be at least `8 + n` bytes — off by exactly 1, a gap
that cannot be closed (you cannot ship a buffer smaller than its own header).

This was verified three ways (`verify_release.rs`, built `--release`):

- **Real deserialize sweep** — `buffer_end = u64::MAX`, sweeping `n` (header
  whitespace padding) and `buffer_len` (0..300 appended bytes):
  `panics=0  accepted(bypass)=0  rejected=2107`, closest `|wrapped - buffer_len| = 1`.
- **Arithmetic Monte Carlo** — 20M random `(buffer_end ≤ u64::MAX, n ≤ 1e8, D)`:
  `WRAP-to-pass = 0`.
- **Release Python binding** (`maturin develop --release`) — the exact input
  that panics in debug now returns a normal
  `SafetensorError: incomplete metadata, file not fully covered` (no
  `PanicException`).

So the impact is confined to `overflow-checks` builds (a controlled panic);
release wheels reject uniformly with no bypass and no downstream out-of-bounds.

## Defense-in-depth note (case 4)

A single `shape=[u64::MAX]` `I64` tensor is rejected earlier by
`ValidationOverflow` (`nelements * bitsize` overflow in `validate()`), before
the buffer check — so that avenue to a large `buffer_end` is closed
independently.

## Trigger / test files

- `trigger.py` — four crafted files: cumulative `buffer_end == 2^64-1` (0 and
  +100 data bytes), offset `= 2^64` (serde reject), and `shape=[u64::MAX]`
  (validate reject). Prints the wrapping arithmetic and each result.
- `trigger_rust.rs` — builds the `buffer_end == 2^64-1` file and calls
  `SafeTensors::deserialize`; run under `cargo run` (debug: panic) and
  `cargo run --release` (graceful reject). Standalone-crate setup as in the
  other Rust triggers (dep `safetensors = { path = ".../safetensors" }`).
- `verify_release.rs` — the release-mode verification harness: a real
  deserialize sweep at `buffer_end = u64::MAX` plus a 20M-sample arithmetic
  Monte Carlo, asserting no wrap-to-pass and no panic. Run with
  `cargo run --release` (same standalone-crate setup).

## Suggested fix (not applied)

Use checked arithmetic at `tensor.rs:420`, e.g. compare via
`buffer_end.checked_add(N_LEN).and_then(|x| x.checked_add(n)) != Some(buffer_len)`
(treating overflow as `MetadataIncompleteBuffer`), or rearrange to
`buffer_len - N_LEN - n` with an underflow guard. This makes every build
profile reject uniformly instead of panicking under `overflow-checks`.
