# safetensors security audit

Working notes and reproducers from a security study of the `safetensors`
library (core Rust crate + PyO3 bindings + Python framework loaders).

Each confirmed issue has its own subfolder containing a `record.md` and the
trigger / test files used to demonstrate it.

## Headline

Against the primary threat (arbitrary code execution at model-load time),
the library is sound: no pickle, no `eval`, no custom operators/layers, no
code path that interprets header/metadata as anything but data. The load
path is bounds-checked slicing over a byte buffer. No memory-safety issue
or ACE vector was found. The confirmed issues below are DoS-class panics in
the slicing arithmetic; the remaining tracked items are lower-severity or
inherent-to-any-weight-format concerns.

## Confirmed issues (with reproducers)

| Folder | Issue | Reachable from | Severity |
|---|---|---|---|
| `slice-reversed-bounds-overflow/` | Reversed slice bounds (`start > stop`) underflow in `slice_byte_ranges` → panic | Python `get_slice[hi:lo]` **and** Rust `view.slice(10..3)` | Low (DoS / crash) |
| `slice-usize-max-bound-overflow/` | Slice bound at `usize::MAX` overflows `narrow_bounds` `*s + 1` → panic | Rust inclusive range `view.slice(..=usize::MAX)` (not Python) | Low (DoS / crash) |
| `buffer-length-overflow/` | `buffer_end + N_LEN + n` unchecked add overflows → panic in `overflow-checks` builds (no logic bypass; release wheels reject gracefully) | `deserialize()`/`load()` via crafted offsets summing to `usize::MAX` | Low (DoS in debug/overflow-checks builds only) |
| `dos-vectors/` | Metadata amplification: valid ~100 MB header (~1.3M tensors) → ~11–25 s + ~1–2 GB RAM (~13–19× file). Other DoS ideas mitigated. | `deserialize()`/`load_file()` | Low–med (resource exhaustion, no tiny-file amplification) |
| `weight-backdoor/` | Trigger→target backdoor encoded entirely in weights; valid, scanner-clean file; normal-input behavior bit-for-bit identical to a clean model | any normal `load_file`/`load_model` | High impact, but by design (format is safe; model behavior is not a format property) |

Both were found by the `fuzz_slice` cargo-fuzz target added in this branch
(`safetensors/fuzz/fuzz_targets/fuzz_slice.rs`). Root cause is shared: the
range check at `slice.rs:363` (`start >= dim || stop > dim`) runs *after* the
bound arithmetic and never enforces `start <= stop`.

## Open investigations (tracked, no reproducer yet)

These are on the to-do list but not yet exercised with test files:

- **Mutable-bytearray aliasing / BOOL byte quirk** in `deserialize`/`load`.
- **`load_model` `__metadata__` key remapping** review.
- **Polyglot / trailing-data scanner evasion** (expected: rejected by
  `MetadataIncompleteBuffer`).

## Environment to reproduce

```
source /home/user/safetensors/.venv/bin/activate   # numpy + torch installed
# Rust: stable + nightly + cargo-fuzz installed
```
