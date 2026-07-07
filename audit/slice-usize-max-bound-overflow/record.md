# Slice bound at `usize::MAX` overflows `narrow_bounds`

- **Component:** core crate — `safetensors/src/slice.rs` (`narrow_bounds`)
- **Reachable from:** Rust `TensorView::slice(..=usize::MAX)` (inclusive range). **Not** reachable from the Python bindings.
- **Class:** integer-overflow panic → Denial of Service / unexpected crash
- **Severity:** Low (requires an extreme bound value; controlled panic, no memory unsafety)
- **Found by:** `fuzz_slice` cargo-fuzz target (`safetensors/fuzz/fuzz_targets/fuzz_slice.rs`)

## Summary

`narrow_bounds` adds 1 to a bound without checking for overflow:

```
slice.rs:317   let start = match left {
slice.rs:318       Bound::Unbounded => 0,
slice.rs:319       Bound::Included(s) => *s,
slice.rs:320       Bound::Excluded(s) => *s + 1,     // overflows if s == usize::MAX
slice.rs:322   let stop = match right {
slice.rs:323       Bound::Unbounded => dim,
slice.rs:324       Bound::Included(s) => *s + 1,      // overflows if s == usize::MAX
slice.rs:325       Bound::Excluded(s) => *s,
```

A bound of `usize::MAX` makes `*s + 1` wrap. With `overflow-checks` on this
panics; in release it wraps to `0` and then the reversed/degenerate range is
mishandled downstream.

## Reachability

- **Rust:** `impl_from_range!(RangeToInclusive<usize>)` / `RangeInclusive<usize>`
  map the end bound to `Bound::Included(idx)`, so `view.slice(..=usize::MAX)`
  hits `slice.rs:324`. A manually built `TensorIndexer::Narrow(Bound::Excluded(usize::MAX), ..)`
  hits `slice.rs:320`.
- **Python:** NOT reachable. `parse_indexers` only ever emits `Included` start
  bounds and `Excluded` stop bounds (never `Excluded` start / `Included` stop),
  so neither `+1` branch is exercised from `get_slice(...)[...]`.

This is therefore a robustness issue for direct consumers of the Rust crate,
not the Python model-loading path.

## Trigger / test files

- `trigger_rust.rs` — `view.slice(..=usize::MAX)`. Build as a standalone crate
  (see the header comment in the file). Debug build panics at `slice.rs:324`.
- `../../safetensors/fuzz/fuzz_targets/fuzz_slice.rs` — the fuzz target that
  discovered the sibling `Excluded(usize::MAX)` case at `slice.rs:320`.

## Observed output (Rust standalone, debug)

```
thread 'main' panicked at src/slice.rs:324:31: attempt to add with overflow
```

## Suggested fix (not applied)

Use `checked_add(1)` in `narrow_bounds` and saturate to `dim` (an out-of-range
bound is already an error / clamp), or resolve bounds with the same
empty-selection handling proposed for the reversed-bounds issue.
